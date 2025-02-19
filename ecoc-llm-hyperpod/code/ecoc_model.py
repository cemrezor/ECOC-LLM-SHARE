import torch
import torch.nn as nn
from torch.nn import functional as F
import json
import numpy as np

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
  def __init__(self, config, device='cpu'):
        super().__init__()
        self.config = config 
        self.device = device
        self.block_size = config.block_size
        self.embedings = nn.Embedding(config.vocab_size, config.n_embed)
        self.position_embedings = nn.Embedding(config.max_pos_n_embed, config.n_embed)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_final = nn.LayerNorm(config.n_embed)

        # [ECOC specific change]
        token_to_ecoc_map, ecoc_bits = self.generate_ecoc_codewords(config.vocab_size)
        self.ecoc_head = nn.Linear(config.n_embed, ecoc_bits)
        
        # [TODO] opmized to retrieve torch tensor for ecoc inde. Check this later !!!!
        self.ecoc_target_tensor = torch.tensor(
            [token_to_ecoc_map[token] for token in range(config.vocab_size)], dtype=torch.float32
        ).to(self.device)


  def token_id_to_ecoc(self, token_id):
    return self.ecoc_target_tensor[token_id]

  def ecoc_to_token_ids_3d(self, targets):   
    batch_size, seq_length, _ = targets.shape
    tokens = torch.zeros((batch_size, seq_length), dtype=torch.long, device=targets.device)

    for i in range(batch_size):
      for j in range(seq_length):
        vec = targets[i, j] 
        exact_matches = (self.ecoc_target_tensor == vec).all(dim=1)
        if exact_matches.any():
          token_id = torch.nonzero(exact_matches, as_tuple=True)[0].item()
          tokens[i, j] = token_id
        else:
          raise ValueError("Non existing ecoc code !!!")
    return tokens


  def generate_ecoc_codewords(self, vocab_size, r=0, seed=42):
    np.random.seed(seed)
    log2_v = int(np.ceil(np.log2(vocab_size))) 
    ecoc_bits = log2_v + r

    binary_codes = [format(i, f'0{log2_v}b') for i in range(vocab_size)]
    binary_matrix = np.array([[int(bit) for bit in code] for code in binary_codes])

    if r > 0:
        random_bits = np.random.randint(0, 2, (vocab_size, r))  
        binary_matrix = np.hstack((binary_matrix, random_bits))
    token_to_ecoc_map = {i: binary_matrix[i] for i in range(vocab_size)}

    return token_to_ecoc_map, ecoc_bits 
   

  def get_parameters(self):
    return sum(p.numel() for p in self.parameters())

  def save(self, path):
      checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": vars(self.config)
        }

      torch.save(checkpoint, path)

  def forward(self, idx, targets=None):
    B, T = idx.shape
    token_embed = self.embedings(idx) # (B, T, C)
    position_embed = self.position_embedings(torch.arange(T,  device=self.device)) # (T, C)
    x = token_embed + position_embed # (B, T, C)
    x = self.dropout(x) # (B, T, C)
    x = self.blocks(x) # (B, T, C)
    x = self.ln_final(x) # (B, T, C)
    
    logits = self.ecoc_head(x)  # (B, T, ecoc_bits)

    if targets is None:
        aligned_targets = None
        loss = None
    else:
        logits = logits[:, :-1, :] 
        shifted_targets = targets[:, 1:]
        aligned_targets = self.ecoc_target_tensor[shifted_targets].contiguous()
        
        loss = F.binary_cross_entropy_with_logits(logits, aligned_targets.float())

    return logits, aligned_targets, loss



  # def decode_ecoc_predictions_from_logits(self, ecoc_logits, cutoff=0.5):

  #     probabilities = torch.sigmoid(ecoc_logits)
  #     pred_ecoc_codes = (probabilities > cutoff).int().detach().cpu().numpy()

  #     batch_size, seq_length, _ = pred_ecoc_codes.shape
  #     pred_tokens = torch.zeros((batch_size, seq_length), dtype=torch.long, device=self.device)  # Output tensor

  #     for i in range(batch_size):
  #       for j in range(seq_length):
  #          pred_bit_vector = pred_ecoc_codes[i, j]
  #          distances = torch.sum((self.ecoc_target_tensor - pred_bit_vector) ** 2, dim=1) 

  #          closest_token = torch.argmin(distances).item() 
  #          pred_tokens[i, j] = closest_token

  #     return torch.tensor(pred_tokens, device=self.device)

  def decode_ecoc_predictions_topk_from_logits(self, ecoc_logits, threshold=0.5, top_k=1):
      batch_size, sequence_length, ecoc_bits = ecoc_logits.shape

      probabilities = torch.sigmoid(ecoc_logits)
      predicted_bits = (probabilities > threshold).int()

      predicted_bits_float = predicted_bits.view(batch_size * sequence_length, ecoc_bits).float()
      
      target_tensor_float = self.ecoc_target_tensor.float()
      expanded_predicted_bits = predicted_bits_float.unsqueeze(1)
      expanded_target_tensor = target_tensor_float.unsqueeze(0)

      diffs = (expanded_predicted_bits - expanded_target_tensor) ** 2
      distances = diffs.sum(dim=-1)

      neg_distances = -distances
      top_k_indices = torch.topk(neg_distances, k=top_k, dim=1).indices
      top_k_tokens = top_k_indices.view(batch_size, sequence_length, top_k)

      return top_k_tokens

  def generate(self, idx, max_tokens, temperature=1.0, top_k=None):
      pass 
      # for _ in range(max_tokens):
      #   idx_cond = idx[:, -self.block_size:] 
      #   logits, _, _ = self(idx_cond) 

      #   logits = logits[:, -1, :]
      #   logits = logits / temperature  

      #   if top_k is not None:
      #         v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
      #         logits[logits < v[:, [-1]]] = -float("Inf")

      #   decoded_tokens = self.decode_ecoc_predictions(logits)
      #   idx_next = decoded_tokens.unsqueeze(-1) 

      #   idx = torch.cat((idx, idx_next), dim=1)   
      # return idx
