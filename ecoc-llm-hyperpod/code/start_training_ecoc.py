import sys
import argparse
import logging
import torch
import wandb
from config import config
from utils import load_data
from trainer import Trainer
from tokenizer import Tokenizer
from models.ecoc_min_model import MinimalEcocGPT2
from models.ecoc_ova_model import OvaECOCGPT2
import math
import numpy as np



logger = logging.getLogger(__name__)
def initialize_wandb(model_config, run_name):
    wandb.login(key=config["wandb"]["key"])
    return wandb.init(
        project=config["wandb"]["project"], 
        name=run_name,
        config={
            "epochs": model_config["epochs"],
            "batch_size": model_config["batch_size"],
            "learning_rate": model_config["lr"],
            "block_size": model_config["block_size"],
            "vocab_size": model_config["vocab_size"],
            "r": model_config["r"]
        }
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train GPT model on TinyStories (or another dataset).")
    parser.add_argument("--model-config", type=str, required=True, 
                        help="Which config key to load (e.g. gpt-1M, gpt-15M, etc.)")
    parser.add_argument("--ecoc-type", type=str, required=True, 
                        help="Which ecoc model to run.", default="minimal")
    return parser.parse_args()

def get_model(ecoc_type, model_config, device):
    if ecoc_type == "minimal":
        model = MinimalEcocGPT2(model_config, device=device)
    elif ecoc_type == "ova":
        model = OvaECOCGPT2(model_config, device=device)
    else:
        raise ValueError(f"Invalid ECOC type: {ecoc_type}. Must be one of ['minimal', 'ova']")
    return model

def main():
    args = parse_args()
    model_config = config[args.model_config]
    logger.info("Using Model Config: %s", model_config)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info("Using device: %s", device)

    train_data, val_data = load_data(
        config["data"],
        batch_size=model_config["batch_size"],
        n=model_config["n"],
        device=device
    )

    tokenizer = Tokenizer(
        config["tokenizer"],
        k=model_config["vocab_size"] - 1,
        file_path="tokens.json",
        device=device
    )

    model = get_model(args.ecoc_type, model_config, device)
             
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=model_config["lr"])

    run_name = f"{config['wandb']['prefix']}-ecoc-{args.ecoc_type}-training-{args.model_config}-vocab-{model_config['vocab_size']}-random-{model_config['r']}-epochs-{model_config['epochs']}"

    wandb_run = initialize_wandb(model_config, run_name)

    trainer = Trainer(
        model_config=model_config,
        model=model,
        optimizer=optim,
        train_data=train_data,
        val_data=val_data,
        encoder=tokenizer.encoder,
        wandb_run=wandb_run,
        device=device
    )

    trainer.train(checkpoint_path=f"{config['checkpoints']['location']}", run_name=run_name)

    logger.info("ECOC Training completed successfully.")


if __name__ == "__main__":
    main()