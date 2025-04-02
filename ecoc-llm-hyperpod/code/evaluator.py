import torch
import math
from config import config


def calculate_perplexity(avg_loss):
    return math.exp(avg_loss) if avg_loss < 100 else float('inf')

def calculate_top_k_accuracy(logits, aligned_targets, model, k=5, eos=False, encoded_entry = None):
        
    if hasattr(model, "ecoc_head") or hasattr(model, "linear_heads"):
        top_k_tokens = model.ecoc_logits_to_topk_tokens_3d(
            ecoc_logits=logits,
            top_k=k
        )
        EOS = model.config.vocab_size-1
        gt_tokens = model.ecoc_to_token_ids_3d(aligned_targets)
        batch_size, seq_length = gt_tokens.shape
        gt_tokens_expanded = gt_tokens.unsqueeze(-1)  # (B, T, 1)
        correct_mask = (top_k_tokens == gt_tokens_expanded).any(dim=-1)  # (B, T)
        correct_count = correct_mask.sum().item()
        total = batch_size * seq_length

        if eos == True:
            EOS_masker = gt_tokens.clone().detach()
            EOS_prev = encoded_entry[:, :-1]
            EOS_mask = (EOS_masker!=EOS)&(EOS_prev!=EOS)
                        
            correct_mask_eos = correct_mask&EOS_mask
            correct_count_eos = correct_mask_eos.sum().item()
            total_eos = EOS_mask.sum().item() 
            top_k_acc_EOS_masked = correct_count_eos / total_eos
            return top_k_acc_EOS_masked

        else:
            return correct_count / total


    else :    
        probabilities = torch.softmax(logits, dim=-1)
        top_k_preds = torch.topk(probabilities, k=5, dim=-1).indices
        print(f"aligned_targets shape: {aligned_targets.shape}")
        print(f"top_k_preds shape: {top_k_preds.shape}")
        correct_predictions = (aligned_targets.unsqueeze(-1) == top_k_preds).any(dim=-1)

        if eos == True:
            EOS_masker = aligned_targets.clone().detach()
            EOS_pre = encoded_entry[:, :-1]            
            EOS_mask = (EOS_masker!=EOS)&(EOS_pre!=EOS)
            correct_predictions_eos = correct_predictions&EOS_mask
            correct_count = correct_predictions_eos.sum().item()
            total_per_batch_eos = EOS_mask.sum().item()
            top_k_acc_EOS_masked = correct_count / total_per_batch_eos
            return top_k_acc_EOS_masked
        
        else:
            correct_per_batch = correct_predictions.sum().item()
            total_per_batch = aligned_targets.numel()

            return correct_per_batch / total_per_batch
