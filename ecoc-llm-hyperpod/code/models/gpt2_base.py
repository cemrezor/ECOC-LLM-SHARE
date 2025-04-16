import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)

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


class GPT2Base(nn.Module):
    def __init__(self, config, time, device='cpu'):
        super().__init__()
        self.config = config
        self.device = device
        self.time=time
        self.block_size = config.block_size
        self.embedings = nn.Embedding(config.vocab_size, config.n_embed)
        self.position_embedings = nn.Embedding(config.max_pos_n_embed, config.n_embed)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_final = nn.LayerNorm(config.n_embed)

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

    def forward_gpt2_base(self, idx):
        B, T = idx.shape
        token_embed = self.embedings(idx)
        position_embed = self.position_embedings(torch.arange(T, device=self.device))
        x = token_embed + position_embed
        x = self.dropout(x)
        x = self.blocks(x)
        x = self.ln_final(x)
        return x
    
    def get_parameters(self):
      return sum(p.numel() for p in self.parameters())
    
    def save(self, path):
      checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": vars(self.config)
        }

      torch.save(checkpoint, path)