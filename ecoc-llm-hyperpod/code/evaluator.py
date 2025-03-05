import torch
import math

def calculate_perplexity(avg_loss):
    return math.exp(avg_loss) if avg_loss < 100 else float('inf')

def calculate_top_k_accuracy(logits, aligned_targets, model, k=5):

    if hasattr(model, "ecoc_head"):
        top_k_tokens = model.ecoc_logits_to_topk_tokens_3d(
            ecoc_logits=logits,
            top_k=k
        )
        gt_tokens = model.ecoc_to_token_ids_3d(aligned_targets)
        batch_size, seq_length = gt_tokens.shape
        gt_tokens_expanded = gt_tokens.unsqueeze(-1)  # (B, T, 1)
        correct_mask = (top_k_tokens == gt_tokens_expanded).any(dim=-1)  # (B, T)
        correct_count = correct_mask.sum().item()
        total = batch_size * seq_length

        return correct_count / total


    else :    
        probabilities = torch.softmax(logits, dim=-1)
        top_k_preds = torch.topk(probabilities, k=k, dim=-1).indices
        correct_predictions = (aligned_targets.unsqueeze(-1) == top_k_preds).any(dim=-1)
        correct_per_batch = correct_predictions.sum().item()
        total_per_batch = aligned_targets.numel()

        return correct_per_batch / total_per_batch
