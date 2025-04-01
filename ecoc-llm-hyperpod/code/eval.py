import os
import re
import sys
import argparse
import logging

import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import load_from_disk

from models.ecoc_min_model import MinimalEcocGPT2
from models.ecoc_ova_model import OvaECOCGPT2
from models.softmax_model import SoftmaxGPT2
from tokenizer import Tokenizer
from evaluator import calculate_top_k_accuracy

import wandb

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

def parse_ecoc_type_from_path(checkpoint_path: str) -> str:
  
    basename = os.path.basename(checkpoint_path).lower()

    if "minimal" in basename:
        ecoc_type = "minimal"
    elif "ova" in basename:
        ecoc_type = "ova"
    elif "softmax" in basename:
        ecoc_type = "softmax"
    else:
        # fallback if no substring is found
        logger.warning("No explicit 'minimal', 'ova', or 'softmax' found in checkpoint name. Defaulting to 'softmax'.")
        ecoc_type = "softmax"

    logger.info(f"parse_ecoc_type_from_path => {ecoc_type}")
    return ecoc_type


def load_model_from_checkpoint(checkpoint_path: str, device='cpu'):
    # Step 1: parse ecoc_type from path
    ecoc_type = parse_ecoc_type_from_path(checkpoint_path)

    # Step 2: load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if "config" not in checkpoint:
        raise ValueError("Checkpoint missing 'config' dictionary! Make sure you saved it with model.save(path).")

    saved_config_dict = checkpoint["config"]
    
    if not isinstance(saved_config_dict, dict):
        raise ValueError("checkpoint['config'] is not a dictionary of hyperparams.")

    # Step 3: pick the model class
    if ecoc_type == "minimal":
        model_class = MinimalEcocGPT2
    elif ecoc_type == "ova":
        model_class = OvaECOCGPT2
    else:
        model_class = SoftmaxGPT2

    logger.info(f"Instantiating model_class={model_class.__name__} on device={device} with config keys={list(saved_config_dict.keys())}")

    # Step 4: create the model. 
    #   - We pass 'saved_config_dict' as the config, plus device.
    model = model_class(saved_config_dict, device=device)

    # load state_dict
    if "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint missing 'model_state_dict'!")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, ecoc_type, saved_config_dict

def evaluate_model(
    checkpoint_path: str,
    data_path: str,
    device='cuda',
    num_samples=None,
    batch_size=None,
    wandb_run=None,
    run_name=None
):
    # parse ecoc_type from filename, load model & config from checkpoint
    model, ecoc_type, config_dict = load_model_from_checkpoint(checkpoint_path, device=device)

    # if no run_name, we default to checkpoint's basename minus .bin
    if run_name is None:
        basefile = os.path.basename(checkpoint_path)
        run_name = os.path.splitext(basefile)[0]

    # possibly override batch_size from CLI
    if batch_size is not None:
        config_dict["batch_size"] = batch_size
    else:
        batch_size = config_dict.get("batch_size", 32)

    logger.info(f"Evaluating with run_name={run_name}, ecoc_type={ecoc_type}, batch_size={batch_size}")
    logger.info(f"Dataset path={data_path}; Using device={device}")

    # Prepare tokenizer
    # e.g. we assume 'config_dict' has 'vocab_size' 
    vocab_size = config_dict.get("vocab_size", 3000)
    EOS = vocab_size - 1

    from tokenizer import Tokenizer
    tokenizer = Tokenizer(
        # or you might have your own config key for this
        "tokenizer",
        k=vocab_size - 1,
        file_path="tokens.json",
        device=device
    )

    # Load dataset
    dataset = load_from_disk(data_path)
    if num_samples is not None:
        dataset = dataset.select(range(num_samples))

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size)

    # Evaluate loop
    results = []
    for batch in tqdm(dataloader, total=len(dataloader)):
        encoded_entry = tokenizer.encoder(batch['text'], padding=True, truncation=True)
        logits, aligned_targets, loss = model(encoded_entry, encoded_entry)

        if hasattr(model, "ecoc_head"):
            # ECOC-based model
            top_k_tokens = model.ecoc_logits_to_topk_tokens_3d(ecoc_logits=logits, top_k=5)
            gt_tokens = model.ecoc_to_token_ids_3d(aligned_targets)

            # mask out EOS
            EOS_masker = gt_tokens.clone().detach()
            EOS_pre = encoded_entry[:, :-1]
            EOS_mask = (EOS_masker != EOS) & (EOS_pre != EOS)

            bsz_cur, seq_len_cur = gt_tokens.shape
            gt_tokens_expanded = gt_tokens.unsqueeze(-1)
            correct = (top_k_tokens == gt_tokens_expanded).any(dim=-1)
            correct_masked = correct & EOS_mask
            correct_count = correct_masked.sum().item()
            total = EOS_mask.sum().item()
            top_k_acc_eos_masked = correct_count / total if total>0 else 0.0

            # overall top_k
            from evaluator import calculate_top_k_accuracy
            top_k_acc = calculate_top_k_accuracy(logits, aligned_targets, model, k=5)
            results.append([top_k_acc, top_k_acc_eos_masked])

        else:
            # softmax-based
            from evaluator import calculate_top_k_accuracy
            probabilities = torch.softmax(logits, dim=-1)
            top_k_preds = torch.topk(probabilities, k=5, dim=-1).indices

            # mask out EOS
            EOS_masker = aligned_targets.clone().detach()
            EOS_pre = encoded_entry[:, :-1]
            EOS_mask = (EOS_masker != EOS) & (EOS_pre != EOS)

            correct_predictions = (aligned_targets.unsqueeze(-1) == top_k_preds).any(dim=-1)
            correct_masked = correct_predictions & EOS_mask
            correct_count = correct_masked.sum().item()
            total = EOS_mask.sum().item()
            top_k_acc_eos_masked = correct_count / total if total>0 else 0.0

            top_k_acc = calculate_top_k_accuracy(logits, aligned_targets, None, k=5)
            results.append([top_k_acc, top_k_acc_eos_masked])

    # Summarize
    df = pd.DataFrame(results, columns=['Top_k_accuracy', 'Top_k_EOS_masked'])
    top_k_acc_mean = df['Top_k_accuracy'].mean()
    top_k_acc_eos_masked_mean = df['Top_k_EOS_masked'].mean()

    logger.info(f"[Final] top-5 accuracy (mean): {top_k_acc_mean:.4f}")
    logger.info(f"[Final] top-5 EOS_masked (mean): {top_k_acc_eos_masked_mean:.4f}")

    if wandb_run is not None:
        wandb_run.log({
            "eval/ecoc_type": ecoc_type,
            "eval/run_name": run_name,
            "eval/top_k_acc_mean": top_k_acc_mean,
            "eval/top_k_acc_eos_masked_mean": top_k_acc_eos_masked_mean
        })

    return top_k_acc_mean, top_k_acc_eos_masked_mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to .bin checkpoint with naming indicating ecoc_type")
    parser.add_argument("--data_path", type=str, required=True, help="Path to HF dataset on disk")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--num_samples", type=int, default=None, help="Optionally limit # of samples for debugging")
    parser.add_argument("--batch_size", type=int, default=None, help="Override config batch size if provided")
    parser.add_argument("--run_name", type=str, default=None, help="Name for logging/wandb")
    parser.add_argument("--use_wandb", action="store_true", help="If set, logs final results to wandb")
    args = parser.parse_args()

    wandb_run = None
    if args.use_wandb:
        if wandb is None:
            raise ImportError("wandb not installed – remove --use_wandb or install wandb.")
        ecoc_type = parse_ecoc_type_from_path(args.checkpoint_path)
        run_name = f"eval-{args.checkpoint_path}"

        wandb_run = wandb.init(
            project="Pretraining_ECOC_LLMs",
            name=run_name,
            config={"ecoc_type": ecoc_type}
        )

    evaluate_model(
        checkpoint_path=args.checkpoint_path,
        data_path=args.data_path,
        device=args.device,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        wandb_run=wandb_run,
        run_name=args.run_name
    )

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()