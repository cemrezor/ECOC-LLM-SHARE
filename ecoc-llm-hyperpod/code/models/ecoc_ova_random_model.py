import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import logging
import sys

from models.gpt2_base import GPT2Base

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

def create_ova_plus_random_codebook(vocab_size: int, extra_bits: int, seed: int = 42):
    """
    Builds an (vocab_size, vocab_size + extra_bits) codebook.
    The first vocab_size columns are a one-hot OVA code,
    and the last extra_bits columns are random bits.
    """
    codebook = torch.zeros(vocab_size, vocab_size, dtype=torch.float32)
    for i in range(vocab_size):
        codebook[i, i] = 1.0

    rng = np.random.default_rng(seed)
    random_bits_np = rng.integers(0, 2, size=(vocab_size, extra_bits), endpoint=False)
    random_bits = torch.tensor(random_bits_np, dtype=torch.float32)

    extended_codebook = torch.cat([codebook, random_bits], dim=1)
    return extended_codebook


class OvaPlusRandomECOCGPT2(GPT2Base):
    """
    OVA-based ECOC with an additional 'extra_bits' random bits for each token.
    total ecoc_bits = vocab_size + extra_bits
    """
    def __init__(self, config, device='cpu', time=False):
        super().__init__(config, device=device, time=time)
        
        self.extra_bits = config.extra_bits
        self.ecoc_bits = config.vocab_size + self.extra_bits
        
        self.ecoc_target_tensor = create_ova_plus_random_codebook(
            vocab_size=config.vocab_size,
            extra_bits=self.extra_bits,
            seed=42
        ).to(self.device)

        self.ecoc_head = nn.Linear(config.n_embed, self.ecoc_bits)
        self.logger = logging.getLogger(__name__)
        self.logger.info(
            f"[OvaPlusRandomECOCGPT2] Initialized with {self.ecoc_bits} bits "
            f"({config.vocab_size} OVA + {self.extra_bits} random)."
        )

    def forward(self, idx, targets=None):
        """
        Forward pass. shape => (B, T, ecoc_bits).
        If targets given, compute BCE with respect to extended OVA+random code.
        """
        x = self.forward_gpt2_base(idx)
        
        start_t0 = time.process_time()
        logits = self.ecoc_head(x)  # shape => (B, T, vocab_size + extra_bits)
        if self.time==True:
            self.logger.info("Time taken between t=0 to t=1", time.process_time() - start_t0)

        if targets is None:
            aligned_targets = None
            loss = None
        else:
            logits = logits[:, :-1, :]        # shape => (B, T-1, vocab_size+extra_bits)
            shifted_targets = targets[:, 1:]  # shape => (B, T-1)

            aligned_targets = self.ecoc_target_tensor[shifted_targets].contiguous()

            start_t2 = time.process_time()
            loss = F.binary_cross_entropy_with_logits(
                logits, aligned_targets.float()
            )
            if self.time==True:
                self.logger.info("Time taken between t=2 to t=3", time.process_time() - start_t2)
            
        return logits, aligned_targets, loss

    def ecoc_logits_to_topk_tokens_3d(self, ecoc_logits, top_k=1):
        """
        For inference: We'll still do an OVA-like decode from the first 'vocab_size' bits,
        ignoring random bits. e.g. we can do:
        shape => (B, T, vocab_size+extra_bits)
        We'll select the first 'vocab_size' portion for picking classes.
        """
        ova_logits = ecoc_logits[..., :self.config.vocab_size]  # (B, T, vocab_size)
        probabilities = torch.sigmoid(ova_logits)               # in [0,1]

        top_k_preds = torch.topk(probabilities, k=top_k, dim=-1).indices
        return top_k_preds  # shape => (B, T, top_k)
    
    def ecoc_to_token_ids_3d(self, targets: torch.Tensor) -> torch.Tensor:
        """
        Converts a batch of extended OVA+random bit vectors into token IDs.
        
        'targets' has shape (B, T, vocab_size+extra_bits),
        the first 'vocab_size' columns are OVA bits, and
        the next 'extra_bits' columns are random bits.
        
        We simply take argmax across the OVA portion (the first vocab_size columns)
        to identify which token ID each row corresponds to.
        
        Returns a shape (B, T) with the token IDs.
        """
        # 1) Slice out the first vocab_size columns => shape (B, T, vocab_size)
        ova_part = targets[..., :self.config.vocab_size]

        # 2) argmax across dim=-1 to find which bit is 1
        # shape => (B, T)
        token_ids = ova_part.argmax(dim=-1)

        return token_ids

    def generate(self, idx, max_tokens=20):
        """
        Same autoregressive logic. We'll do top-1 from OVA region only.
        shape => (B, T + max_tokens)
        """
        for _ in range(max_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _, _ = self(idx_cond)
            # shape => (B, T_cond, vocab_size + extra_bits)

            last_logits = logits[:, -1:, :]   # shape => (B,1,vocab_size+extra_bits)
            ova_portion = last_logits[..., :self.config.vocab_size] 
            probabilities = torch.sigmoid(ova_portion)     # shape => (B,1,vocab_size)
            top1 = torch.topk(probabilities, k=1, dim=-1).indices  # shape => (B,1,1)

            next_token = top1.squeeze(-1)  # shape => (B,1)
            idx = torch.cat((idx, next_token), dim=1)

        return idx
