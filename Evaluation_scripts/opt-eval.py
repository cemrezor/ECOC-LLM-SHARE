import os
import json
import math
import torch
import torch.nn as nn
import bitsandbytes as bnb
from transformers import (
    AutoTokenizer, 
    OPTPreTrainedModel, 
    OPTModel, 
    Trainer,
)
import transformers
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional, Tuple, Union, List
from datasets import load_dataset
from huggingface_hub import login
from peft import LoraConfig, get_peft_model
import wandb
from transformers import OPTForCausalLM
import tqdm
from bert_score import score as bert_score

model = OPTForCausalLM.from_pretrained("linkanjarad/OPT-Alpaca-125M", return_dict=True, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")

data = load_dataset("disham993/alpaca-train-validation-test-split")

def generate_prompt(data_point):
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{data_point["instruction"]}
### Input:
{data_point["input"]}
### Response:
{data_point["output"]}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
{data_point["instruction"]}
### Response:
{data_point["output"]}"""


def generate_predictions_normal(data_split):
    predictions = []
    for data_point in tqdm(data_split, desc="Generating Predictions"):
        prompt = generate_prompt(data_point)
        inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
        inputs = inputs['input_ids']
        
        # Generate outputs with the model
        with torch.no_grad():
            outputs = model.generate(input_ids=inputs, max_new_tokens=50)
        
        # Decode the generated output
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predictions.append(decoded_output.strip())
    
    return predictions

# Function to calculate BERTScore
def calculate_bertscore(predictions, references):
    precision, recall, f1 = bert_score(predictions, references, lang="en", verbose=True)
    return precision.mean().item(), recall.mean().item(), f1.mean().item()

validation_predictions = generate_predictions_normal(data["test"])
validation_references = [data_point["output"] for data_point in data["test"]]

validation_precision, validation_recall, validation_f1 = calculate_bertscore(validation_predictions, validation_references)

print(f"Validation BERTScore for normal OPT - Precision: {validation_precision:.4f}, Recall: {validation_recall:.4f}, F1: {validation_f1:.4f}")
