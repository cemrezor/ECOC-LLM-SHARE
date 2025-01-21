import sys
import argparse
import logging
import torch
import wandb
from config import config
from utils import load_data
from trainer import Trainer
from tokenizer import Tokenizer
from model import GPT2

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train GPT model on TinyStories (or another dataset).")
    parser.add_argument("--model-config", type=str, required=True, 
                        help="Which config key to load (e.g. gpt-1M, gpt-15M, etc.)")
    return parser.parse_args()

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
        }
    )

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
        k=model_config["k"],
        file_path="tokens.json",
        device=device
    )

    model = GPT2(model_config, device=device)
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=model_config["lr"])

    run_name = f"{config['wandb']['prefix']}-training-{args.model_config}-vocab-{model_config['vocab_size']}-epochs-{model_config['epochs']}"
    
    wandb_run = initialize_wandb(model_config, run_name)

    trainer = Trainer(
        config=model_config,
        model=model,
        optimizer=optim,
        train_data=train_data,
        val_data=val_data,
        encoder=tokenizer.encoder,
        wandb_run=wandb_run,
        device=device
    )

    trainer.train(
        eval_interval=model_config["eval_interval"],
        eval_steps=model_config["eval_steps"]
    )

    logger.info("Training completed successfully.")

if __name__ == "__main__":
    main()
