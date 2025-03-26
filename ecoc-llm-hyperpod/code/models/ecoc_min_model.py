import torch
import torch.nn as nn
from torch.nn import functional as F
import json
import numpy as np
import time
from models.gpt2_base import GPT2Base 
from scipy.spatial.distance import cdist

class MinimalEcocGPT2(GPT2Base):
  def __init__(self, config, device='cpu'):
        super().__init__(config, device=device)
        
        token_to_ecoc_map, ecoc_bits = self._generate_ecoc_codewords(config.vocab_size, config.r)
        
        self.ecoc_head = nn.Linear(config.n_embed, ecoc_bits)
        
        self.ecoc_target_tensor = torch.tensor(
            [token_to_ecoc_map[token] for token in range(config.vocab_size)], dtype=torch.float32
        ).to(self.device)

        self.logger.info(f"[Model] MinimalEcocGPT2 initialized with Ecoc bits: {ecoc_bits}")


  def _generate_ecoc_codewords(self, vocab_size, r=0, seed=42):
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

  def forward(self, idx, targets=None):
    x = self.forward_gpt2_base(idx)  
    
    # # t = 0
    # start_t0 = time.process_time()
    logits = self.ecoc_head(x)  # (B, T, ecoc_bits)
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
        #start_t2 = time.process_time()
        loss = F.binary_cross_entropy_with_logits(logits, aligned_targets.float())
        #print("Time taken between t=2 to t=3", time.process_time() - start_t2)
        # t = 3

    return logits, aligned_targets, loss


  def ecoc_logits_to_topk_tokens_3d(self, ecoc_logits, top_k=1):
      batch_size, sequence_length, ecoc_bits = ecoc_logits.shape

      probabilities = torch.sigmoid(ecoc_logits)
      
      probabilities_2d = probabilities.view(batch_size * sequence_length, ecoc_bits).float()
      
      target_tensor_float = self.ecoc_target_tensor.float()
      # expanded_probs = probabilities_2d.unsqueeze(1)
      # expanded_targets = target_tensor_float.unsqueeze(0)
      # probabilities_convert = probabilities_2d.clone()
      # probabilities_convert[probabilities_convert==0]=-1
      # target_convert = target_tensor_float.clone()
      # target_convert[target_convert==0] =-1

      # probs_sum = (expanded_probs**2).sum(dim=-1) # batch_size * sequence_length
      # targets_sum = expanded_targets.sum(dim=-1)  # vocab_size
      # norm_prob = F.normalize(probabilities_2d)
      # norm_target = F.normalize(target_tensor_float)

      # dot_products = probabilities_2d @ target_tensor_float.T  # (batch_size * sequence_length, vocab_size)

      # dot_products = probabilities_convert @ target_convert.T  # (batch_size * sequence_length, vocab_size)

      # similarity = dot_products/(norm_prob @ norm_target.T)
      # similarity = norm_prob @ norm_target.T

      diff_matrix = torch.cdist(probabilities_2d, target_tensor_float, p=1)
      # probabilities_2d_np = probabilities_2d.cpu().numpy()
      # target_tensor_float_np = target_tensor_float.cpu().numpy()
      # diff_matrix = cdist(probabilities_2d_np, target_tensor_float_np, metric='cityblock')
      # diff_matrix_tensor = torch.from_numpy(diff_matrix).cuda()


      # diff_matrix = cdist(probabilities_2d, target_tensor_float, metric='cityblock')

      # distances = (probs_sum + targets_sum  - 2 * dot_products)

      # diffs = (expanded_probs - expanded_targets) ** 2
      # distances = diffs.sum(dim=-1)

      neg_distances = -diff_matrix

      top_k_indices = torch.topk(neg_distances, k=top_k, dim=1).indices

      top_k_tokens = top_k_indices.view(batch_size, sequence_length, top_k)

      return top_k_tokens

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
