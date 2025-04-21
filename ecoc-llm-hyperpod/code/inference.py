import argparse
import torch
import pandas as pd
from tqdm import tqdm
from model import GPT2
from tokenizer import Tokenizer
from utils import *
import datasets
from config import config

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GPT-2 model")
    parser.add_argument("--model_name", type=str, required=True, help="Model name")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--dataset_name", type=str, required=True, help="Path to dataset")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to tokenizer file")
    parser.add_argument("--output_csv", type=str, required=True, help="Output CSV file path")
    return parser.parse_args()

def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model_config = config[args.model_name]
    batch_size = model_config['batch_size']
    EOS = model_config['vocab_size'] - 1
    DEBUG = True
    
    model = GPT2(model_config, device=device)
    checkpoint = torch.load(args.checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    
    tokenizer = Tokenizer(config.tokenizer, k=model_config.vocab_size - 1, file_path=args.tokenizer_path, device=device)
    dataset = load_from_disk(args.dataset_name)
    dataloader = DataLoader(dataset, batch_size)
    
    results = []
    
    for batch in tqdm(dataloader, total=len(dataloader)):
        encoded_entry = tokenizer.encoder(batch['text'], padding=True, truncation=True)
        correct_total = 0
        n_per_entry = 0
        context_length = encoded_entry.shape[1]
        
        for i in range(1, context_length - 1):
            context = encoded_entry[:, :i]
            check = encoded_entry[:, i:i+2]
            reference = encoded_entry[:, i+1]
            prompt_plus_prediction = model.generate(context.cuda(), max_tokens=1, temperature=1, top_k=None)
            predicted_token = prompt_plus_prediction[:, context.shape[1]:]
            next_top_k_tokens = model.next_token(context.cuda(), top_k=5)
            EOS_mask = (check != EOS).all(dim=1)
            correct_predictions = (reference.cuda().unsqueeze(dim=-1) == next_top_k_tokens).any(dim=-1)
            correct_predictions_not_EOS = correct_predictions & EOS_mask
            correct_count = correct_predictions_not_EOS.sum().item()
            correct_total += correct_predictions.sum().item()
            n_per_entry += EOS_mask.sum().item()
            
            if DEBUG:
                print("====")
                print(f"Predicted Tokens: {[tokenizer.decoder(x) for x in next_top_k_tokens.unsqueeze(dim=-1)]}")
                print(f"Reference Token: {tokenizer.decoder(reference.unsqueeze(dim=-1))}")
                print(f"Single Prediction Accuracy: {correct_count}/{EOS_mask.sum().item()}")
                print(f"Overall Prediction Accuracy: {correct_total}/{n_per_entry}")
        
        top_k_acc = correct_total / n_per_entry
        results.append(top_k_acc)
    
    df = pd.DataFrame(results, columns=['Top_k_accuracy'])
    df.to_csv(args.output_csv, index=False)
    print(f"Mean Top-k Accuracy: {df.Top_k_accuracy.mean()}")

if __name__ == "__main__":
    main()