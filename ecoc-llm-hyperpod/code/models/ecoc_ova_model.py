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

class OvaECOCGPT2(GPT2Base):

    def __init__(self, config, device='cpu'):
        super().__init__(config, device)
        self.ecoc_bits = config.vocab_size
        self.ecoc_target_tensor = self._create_ova_codebook(config.vocab_size).to(self.device)
        self.ecoc_head = nn.Linear(config.n_embed, self.ecoc_bits)
        
        self.logger.info(f"ECOC OVA model initialized with {self.ecoc_bits} bits")
        

    def _create_ova_codebook(self, vocab_size: int) -> torch.Tensor:
        codebook = torch.zeros(vocab_size, vocab_size, dtype=torch.float32)
        for i in range(vocab_size):
            codebook[i, i] = 1.0
        return codebook
    
    def forward(self, idx, targets=None):
        x = self.forward_gpt2_base(idx)

        logits = self.ecoc_head(x)

        if targets is None:
            aligned_targets = None
            loss = None
        else:
            logits = logits[:, :-1, :]        
            shifted_targets = targets[:, 1:]  

            aligned_targets = self.ecoc_target_tensor[shifted_targets].contiguous()

            loss = F.binary_cross_entropy_with_logits(logits, aligned_targets.float())
            
        return logits, aligned_targets, loss


    def ecoc_logits_to_topk_tokens_3d(self, ecoc_logits, top_k=1):
        probabilities = torch.sigmoid(ecoc_logits)
        top_k_preds = torch.topk(probabilities, k=top_k, dim=-1).indices
        return top_k_preds

    def ecoc_to_token_ids_3d(self, targets: torch.Tensor) -> torch.Tensor:
        token_ids = targets.argmax(dim=-1)
        return token_ids

