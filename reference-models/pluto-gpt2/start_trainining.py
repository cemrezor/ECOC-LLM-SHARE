import sys
import argparse
from config import config
import torch
from utils import load_data
from trainer import Trainer
from tokenizer import Tokenizer
import wandb
from model import GPT2


import logging
#from utils import load_data

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(description="Train GPT model on TinyStories (or another dataset).")

    parser.add_argument("--model-config", type=str, required=True, 
                        help="Which config key to load (e.g. gpt-1M, gpt-15M, etc.)")

    return parser.parse_args()

def main():
    args = parse_args()
    model_config = config[args.model_config]
    logger.info("Using Model Config ", list(model_config))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_data, val_data = load_data(
        config.data,
        batch_size = model_config["batch_size"],
        n = model_config["n"],
        device = device
    )

    tokenizer = Tokenizer(
        config.tokenizer,
        k=model_config["k"],
        file_path="tokens.json",
        device=device
    )

    model = GPT2(model_config, device=device)
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=model_config.lr)

    run_name = f"training-{args.model_config}-vocab-{model_config.vocab_size}-epochs-{model_config.epochs}"
    wandb.login(key=config["wandb"]["key"])
    wandb.init(
        project=config["wandb"]["project"], 
        name=run_name
    )

    trainer = Trainer(
        model_config, 
        model, 
        optim, 
        train_data, 
        val_data, 
        tokenizer.encoder, 
        wandb
    )

    tracked_losses = trainer.train(
        epochs=model_config.epochs, 
        eval_interval=model_config.eval_interval, 
        eval_steps=model_config.eval_steps
    )

    trainer.finalize()

    model.save(f"{run_name}.bin")

    logger.info("Training completed.")

if __name__ == "__main__":
    main()