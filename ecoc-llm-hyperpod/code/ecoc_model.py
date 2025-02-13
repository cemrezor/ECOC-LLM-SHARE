import torch
import torch.nn as nn
from torch.nn import functional as F

class Head(nn.Module):
  def __init__(self, config, head_size):
    super().__init__()
    self.key = nn.Linear(config.n_embed, head_size, bias=False)
    self.query = nn.Linear(config.n_embed, head_size, bias=False)
    self.value = nn.Linear(config.n_embed, head_size, bias=False)
    self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))  # (T, T)
    self.dropout = nn.Dropout(config.dropout)

  def forward(self, x):
    B, T, C = x.shape
    k = self.key(x) # (B, T, C)
    q = self.query(x) # (B, T, C)
    wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 # (B, T, C) X (B, C, T) --> (B, T, T)
    wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
    wei = F.softmax(wei, dim=-1)
    wei = self.dropout(wei)
    v = self.value(x)  # (B,T,C)
    out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
    return out
  

class MultiHeadAttention(nn.Module):
  def __init__(self, config, head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(config, head_size) for _ in range(config.n_head)])
    self.proj  = nn.Linear(head_size * config.n_head, config.n_embed)
    self.dropout = nn.Dropout(config.dropout)

  def forward(self,x):
    out = torch.concat([h(x) for h in self.heads], dim=-1)
    out = self.dropout(self.proj(out))
    return out
  

class FeedForward(nn.Module):
  def __init__(self, config):
   super().__init__()
   self.layers = nn.Sequential(
        nn.Linear(config.n_embed, 4 * config.n_embed),
        nn.GELU(),
        nn.Linear(4 * config.n_embed, config.n_embed),
        nn.Dropout(config.dropout),
    )

  def forward(self,x):
    return self.layers(x)
  

class Block(nn.Module):
  def __init__(self, config):
    super().__init__()
    head_size = config.n_embed // config.n_head
    self.sa_heads = MultiHeadAttention(config, head_size)
    self.ffwd = FeedForward(config)
    self.ln1 = nn.LayerNorm(config.n_embed)
    self.ln2 = nn.LayerNorm(config.n_embed)

  def forward(self, x):
    x = x + self.sa_heads(self.ln1(x))
    x = x + self.ffwd(self.ln2(x))
    return x
  

class EcocGPT2(nn.Module):
  def __init__(self, config, ecoc_bits, token_to_ecoc_map, device='cpu'):
        super().__init__()
        self.device = device
        self.block_size = config.block_size
        self.embedings = nn.Embedding(config.vocab_size, config.n_embed)
        self.position_embedings = nn.Embedding(config.max_pos_n_embed, config.n_embed)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_final = nn.LayerNorm(config.n_embed)

        # [ECOC specific change]
        self.ecoc_head = nn.Linear(config.n_embed, ecoc_bits)
        self.token_to_ecoc_map = token_to_ecoc_map


  def get_parameters(self):
    return sum(p.numel() for p in self.parameters())

  def save(self, path):
    torch.save(self.state_dict(), path)

  def forward(self, idx, targets=None):
    B, T = idx.shape
    token_embed = self.embedings(idx) # (B, T, C)
    position_embed = self.position_embedings(torch.arange(T,  device=self.device)) # (T, C)
    x = token_embed + position_embed # (B, T, C)
    x = self.dropout(x) # (B, T, C)
    x = self.blocks(x) # (B, T, C)
    x = self.ln_final(x) # (B, T, C)
    
    logits = self.ecoc_head(x)  # (B, T, vocab_size)

    if targets is None:
        aligned_targets = None
        loss = None
    else:
        aligned_targets = torch.stack([torch.tensor(self.token_to_ecoc_map[token.item()]) for token in targets.view(-1)])
        aligned_targets = aligned_targets.view(B, T, -1).to(self.device)

        loss = F.binary_cross_entropy_with_logits(logits, aligned_targets.float())

    return logits, aligned_targets, loss


def decode_ecoc_predictions(ecoc_outputs, token_to_ecoc):
    pred_codes = (ecoc_outputs > 0.5).int() 
    pred_tokens = []

    token_ids = list(token_to_ecoc.keys())
    ecoc_matrix = torch.stack([torch.tensor(token_to_ecoc[token]) for token in token_ids]).to(ecoc_outputs.device)

    for batch in pred_codes:  
        batch_tokens = []
        for pred in batch:
            distances = torch.sum((ecoc_matrix - pred) ** 2, dim=1)  
            closest_token = token_ids[torch.argmin(distances).item()] 
            batch_tokens.append(closest_token)
        pred_tokens.append(batch_tokens)

    return torch.tensor(pred_tokens, device=ecoc_outputs.device)


def generate(self, idx, max_tokens, temperature=1.0, top_k=None):
  
  for _ in range(max_tokens):
    idx_cond = idx[:, -self.block_size:] 
    logits, _, _ = self(idx_cond) 

    logits = logits[:, -1, :]
    logits = logits / temperature  

    decoded_tokens = decode_ecoc_predictions(logits, self.token_to_ecoc)
    idx_next = decoded_tokens.unsqueeze(-1)  

    idx = torch.cat((idx, idx_next), dim=1)
    
  return idx


  # def next_token(self, idx, temperature=1.0, top_k: int = 1):
  #   # idx is (B, T)
  #   idx_cond = idx[:, -self.block_size:]
  #   logits, _, _ = self(idx_cond) # (B, T, C)
  #   logits = logits[:, -1, :]  / temperature # (B, C)

  #   v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
  #   logits[logits < v[:, [-1]]] = -float('Inf')

  #   probs = F.softmax(logits, dim=-1) # Softmax Independently for C dim
  #   next_top_k_tokens = torch.multinomial(probs, num_samples=top_k) # (B, top_k)
  #   return next_top_k_tokens
