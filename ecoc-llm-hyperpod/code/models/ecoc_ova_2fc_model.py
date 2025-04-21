import torch
import torch.nn as nn
from torch.nn import functional as F
import json
import numpy as np
import time
from models.gpt2_base import GPT2Base 
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

class OvaMTLECOCGPT2(GPT2Base):

    def __init__(self, config, device='cpu', time=False):
        super().__init__(config, device, time=time)
        self.ecoc_bits = config.vocab_size
        self.ecoc_target_tensor = self._create_ova_codebook(config.vocab_size).to(self.device)
        # self.ecoc_head = nn.Linear(config.n_embed, self.ecoc_bits)
        self.linear_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.n_embed, 2, bias=False),  # First linear layer
                nn.ReLU(),  # Activation function
                nn.Linear(2, 1, bias=False),  # Second linear layer for a single logit
                nn.Sigmoid()  # Sigmoid to convert logits to probabilities (0 or 1)
            )
            for _ in range(self.ecoc_bits)
        ])
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"ECOC OVA MTL model initialized with {self.ecoc_bits} bits")
        

    def _create_ova_codebook(self, vocab_size: int) -> torch.Tensor:
        codebook = torch.zeros(vocab_size, vocab_size, dtype=torch.float32)
        for i in range(vocab_size):
            codebook[i, i] = 1.0
        return codebook
    
    def forward(self, idx, targets=None):
        x = self.forward_gpt2_base(idx)
        # t = 0
        start_t0 = time.process_time()
        # logits = self.linear_heads(x)
        # hidden_states = logits[0]
        logits = torch.stack([head(x) for head in self.linear_heads]).squeeze(-1)
        logits = logits.permute(1, 2, 0)  # Shape: [batch_size, seq_len, bit_size]
        if self.time==True:
            self.logger.info("Time taken between t=0 to t=1", time.process_time() - start_t0)
        # print("Time taken between t=0 to t=1", time.process_time() - start_t0)
        # t = 1 

        if targets is None:
            aligned_targets = None
            loss = None
        else:
            logits = logits[:, :-1, :]        
            shifted_targets = targets[:, 1:]  

            aligned_targets = self.ecoc_target_tensor[shifted_targets].contiguous()
            # t = 2
            start_t2 = time.process_time()
            loss = F.binary_cross_entropy_with_logits(logits, aligned_targets.float())
            if self.time==True:
                self.logger.info("Time taken between t=2 to t=3", time.process_time() - start_t2)
            # print("Time taken between t=2 to t=3", time.process_time() - start_t2)
            # t = 3
            
        return logits, aligned_targets, loss


    def ecoc_logits_to_topk_tokens_3d(self, ecoc_logits, top_k=1):
        probabilities = torch.sigmoid(ecoc_logits)
        top_k_preds = torch.topk(probabilities, k=top_k, dim=-1).indices
        return top_k_preds

    def ecoc_to_token_ids_3d(self, targets: torch.Tensor) -> torch.Tensor:
        token_ids = targets.argmax(dim=-1)
        return token_ids