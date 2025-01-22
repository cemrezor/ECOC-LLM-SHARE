import torch
import math

def calculate_perplexity(avg_loss):
    return math.exp(avg_loss) if avg_loss < 100 else float('inf')

def calculate_top_k_accuracy(logits, tokens, k=5):
    probabilities = torch.softmax(logits, dim=-1)  
    top_k_preds = torch.topk(probabilities, k=k, dim=-1).indices  
    correct_predictions = (tokens.unsqueeze(-1) == top_k_preds).any(dim=-1)  
    total_correct = correct_predictions.sum().item()
    total_tokens = tokens.numel()

    return total_correct / total_tokens