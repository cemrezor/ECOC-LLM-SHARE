import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.gpt2_base import GPT2Base
import time
import logging

class SoftmaxGPT2(GPT2Base):
  def __init__(self, config, device='cpu', time=False):
    super().__init__(config, device=device, time=time)
    self.lm_head = nn.Linear(config.n_embed, config.vocab_size)
    self.logger = logging.getLogger(__name__)
    self.logger.info(f"Softmax model initialized.")

  def forward(self, idx, targets=None):

    x = self.forward_gpt2_base(idx)    
        # # t = 0
    start_t0 = time.process_time()
    logits = self.lm_head(x)  # (B, T, vocab_size)
    if self.time==True:
      self.logger.info("Time taken between t=0 to t=1: %f", time.process_time() - start_t0)
    # print("Time taken between t=0 to t=1", time.process_time() - start_t0)
    # t = 1 

    if targets is None:
      loss = None
      aligned_targets = None
    else:
      logits = logits[..., :-1, :].contiguous()
      aligned_targets = targets[..., 1:].contiguous()
      # t = 2
      start_t2 = time.process_time()
      loss = F.cross_entropy(logits.view(-1, logits.size(-1)), aligned_targets.view(-1), ignore_index=50256)
      if self.time==True:
        self.logger.info("Time taken between t=2 to t=3", time.process_time() - start_t2)
      # print("Time taken between t=2 to t=3", time.process_time() - start_t2)
      # t = 3


    return logits, aligned_targets, loss 

  def generate(self, idx, max_tokens, temperature=1.0, top_k=None):
    # idx is (B, T)
    for _ in range(max_tokens):
      idx_cond = idx[:, -self.block_size:]
      logits, _, _ = self(idx_cond) # (B, T, C)
      logits = logits[:, -1, :]  / temperature # (B, C)

      if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float('Inf')
        
      probs = F.softmax(logits, dim=-1) # Softmax Independently for C dim
      idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
      idx = torch.concat((idx, idx_next), dim=1) # (B, T+1)
    return idx

  def next_token(self, idx, temperature=1.0, top_k: int = 1):
    # idx is (B, T)
    idx_cond = idx[:, -self.block_size:]
    logits, _, _ = self(idx_cond) # (B, T, C)
    logits = logits[:, -1, :]  / temperature # (B, C)

    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
    logits[logits < v[:, [-1]]] = -float('Inf')

    probs = F.softmax(logits, dim=-1) # Softmax Independently for C dim
    next_top_k_tokens = torch.multinomial(probs, num_samples=top_k) # (B, top_k)
    return next_top_k_tokens
