import sys
from models.gpt2_base import GPT2Base
from models.ecoc_min_model import MinimalEcocGPT2
import torch
from tokenizer import Tokenizer
from utils import *
import datasets
import pandas as pd
from tqdm import tqdm
from config import config
from evaluator import calculate_top_k_accuracy

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_name= "gpt-15M"

model_config = config[model_name]
batch_size = model_config['batch_size']
EOS = model_config['vocab_size']-1
# DEBUG = True

model = GPT2Base(model_config, device=device)
# model = MinimalEcocGPT2(model_config, device=device)


# checkpoint = torch.load('/fsx/ubuntu/ecoc-llm-env/checkpoints/sha-training-gpt-15M-vocab-3000-epochs-1-epoch-1.bin', map_location=torch.device('cpu'))
# checkpoint = torch.load('/fsx/ubuntu/ecoc-llm-env/checkpoints/sha-training-gpt-30M-vocab-3000-epochs-1-epoch-1.bin', map_location=torch.device('cpu'))
# checkpoint = torch.load('/fsx/ubuntu/ecoc-llm-env/checkpoints/manhattan/yu-ecoc-minimal-training-gpt-15M-vocab-1000-random-40-epochs-1-epoch-1.bin', map_location=torch.device('cpu'))
checkpoint = torch.load('/fsx/ubuntu/hragarwl/ecoc-llm-env/checkpoints/sha-training-gpt-15M-vocab-1000-epochs-1-epoch-1.bin', map_location=torch.device('cpu'))

model_state_dict = checkpoint['model_state_dict']

model.load_state_dict(model_state_dict)

# model.load_state_dict(checkpoint, strict=False)


model.to(device)
model.eval()
#print(model)

tokenizer = Tokenizer(config.tokenizer, k=model_config.vocab_size-1, file_path="tokens.json", device=device)

DATASET_NAME = "/fsx/ubuntu/ecoc-llm-env/data/validation"


dataset = load_from_disk(DATASET_NAME)#.select(range(25600))

dataloader = DataLoader(dataset, batch_size)

# unconditional = torch.zeros((1, 1), dtype=torch.long, device=device)
# prompt = "Answer in just one word: What is the capital of India? "

results =[]

for batch in tqdm(dataloader, total=len(dataloader)):
    encoded_entry = tokenizer.encoder(batch['text'], padding=True, truncation=True)
    logits, aligned_targets, loss = model(encoded_entry, encoded_entry)
    top_k_acc = calculate_top_k_accuracy(logits, aligned_targets, model, k=5, eos=True, encoded_entry = encoded_entry)
    print(top_k_acc)
    print(encoded_entry)
    results.append(top_k_acc)
    df = pd.DataFrame(results, columns=['Top_k_accuracy'])

    # if hasattr(model, "ecoc_head"):

    #     top_k_tokens = model.ecoc_logits_to_topk_tokens_3d(
    #         ecoc_logits=logits,
    #         top_k=5
    #     )

    #     gt_tokens = model.ecoc_to_token_ids_3d(aligned_targets)
    #     EOS_masker = gt_tokens.clone().detach()
    #     EOS_prev = encoded_entry[:, :-1]
    #     EOS_mask = (EOS_masker!=EOS)&(EOS_prev!=EOS)
    #     # EOS_mask = (EOS_prev!=EOS) 
    #     # print(top_k_tokens)
    #     # print(gt_tokens)

    #     batch_size, seq_length = gt_tokens.shape
    #     gt_tokens_expanded = gt_tokens.unsqueeze(-1)  
    #     correct = (top_k_tokens == gt_tokens_expanded).any(dim=-1)  
    #     correct_masked = correct&EOS_mask
    #     correct_count = correct_masked.sum().item()
    #     total = EOS_mask.sum().item() 
    #     top_k_acc_EOS_masked = correct_count / total
    #     eos_count = (top_k_tokens == EOS).any(dim=-1).sum().item()
    #     eos_context = (gt_tokens_expanded == EOS).any(dim=-1).sum().item()
    #     eos_perc = eos_count/(batch_size*seq_length)

    #     # print(top_k_tokens)
    #     # print(gt_tokens)
    #     # print(f'gt_eos/pred_eos: {eos_context}/{eos_count}')
    #     top_k_acc = calculate_top_k_accuracy(logits, aligned_targets, model, k=5)

    #     # results.append(top_k_acc)

    #     results.append([top_k_acc, top_k_acc_EOS_masked, eos_perc, eos_count, eos_context])
    #     # print('============')
    #     # print(f'top_k with eos mask: {top_k_acc_EOS_masked:.2f}')
    #     # print(f'top_k: {top_k_acc:.2f}')

    #     df = pd.DataFrame(results, columns=['Top_k_accuracy', 'Top_k_EOS_masked', 'EOS_perc', 'EOS_pred_total', 'EOS_context'])

    
    # else:

    #     EOS_masker = aligned_targets.clone().detach()
    #     EOS_pre = encoded_entry[:, :-1]
    #     EOS_mask = (EOS_masker!=EOS)&(EOS_pre!=EOS)

    #     probabilities = torch.softmax(logits, dim=-1)
    #     top_k_preds = torch.topk(probabilities, k=5, dim=-1).indices
    #     correct_predictions = (aligned_targets.unsqueeze(-1) == top_k_preds).any(dim=-1)
    #     correct_predictions_eos = correct_predictions&EOS_mask
    #     correct_count = correct_predictions_eos.sum().item()
    #     # correct_per_batch_eos = correct_predictions.sum().item()
    #     total_per_batch_eos = EOS_mask.sum().item()
    #     eos_count = (top_k_preds == EOS).any(dim=-1).sum().item()
    #     eos_context = (aligned_targets == EOS).sum().item()

    #     eos_perc = eos_count/(batch_size*aligned_targets.shape[-1])

    #     print(f'gt_eos/pred_eos: {eos_context}/{eos_count}')

    #     top_k_acc_EOS_masked = correct_count / total_per_batch_eos
    #     top_k_acc = calculate_top_k_accuracy(logits, aligned_targets, None, k=5)

    #     # results.append(top_k_acc)

    #     results.append([top_k_acc, top_k_acc_EOS_masked, eos_perc, eos_count, eos_context])
    #     # print('============')
    #     # print(f'top_k with eos mask: {top_k_acc_EOS_masked:.2f}')
    #     # print(f'top_k: {top_k_acc:.2f}')

    
    #     # df = pd.DataFrame(results, columns=['Top_k_accuracy'])
    #     df = pd.DataFrame(results, columns=['Top_k_accuracy', 'Top_k_EOS_masked', 'EOS_perc', 'EOS_pred_total', 'EOS_context'])

top_k_accuracy_mean = df.Top_k_accuracy.mean()
# top_k_accuracy_mean_EOS_masked = df.Top_k_EOS_masked.mean()
# eos_perc_mean = df.EOS_perc.mean()
# eos_total_pred = df.EOS_pred_total.sum()
# eos_total_context=df.EOS_context.sum()

# df.to_csv('top_k_accuracy.csv')

print(f'top_k {top_k_accuracy_mean:.2f}')
# print(f'top_k_eos_masked {top_k_accuracy_mean_EOS_masked:.2f}')
# print(f'eos_perc_mean {eos_perc_mean:.2f}')
# print(f'eos_pred_total {eos_total_pred:.2f}')
# print(f'eos_total_context {eos_total_context:.2f}')