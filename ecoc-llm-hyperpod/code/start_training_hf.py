import sys
import argparse
import logging
import torch
import wandb
from config import config
from config import get_qwen2_hf_config  # import the helper function
from utils import load_data
from trainer_hf import Trainer
from tokenizer import Tokenizer
from models.ecoc_min_model import MinimalEcocGPT2
from models.ecoc_ova_model import OvaECOCGPT2
from models.softmax_model import SoftmaxGPT2
from models.ecoc_ova_2fc_model import OvaMTLECOCGPT2
from transformers import Qwen2Config
# from models.ecoc_min_2fc_model import MinimalEcocMTLGPT2
from models.qwen_minimal_ecoc import Qwen2ForCausalLM
import math
import numpy as np
from models.model_factory import get_model
from huggingface_hub import login
import bitsandbytes as bnb


login(token="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")

logger = logging.getLogger(__name__)

def initialize_wandb(model_config, run_name):
    wandb.login(key=config["wandb"]["key"])
    return wandb.init(
        project=config["wandb"]["project"], 
        name=run_name,
        config={
            "epochs": model_config.epochs,
            "batch_size": model_config.batch_size,
            "learning_rate": model_config.lr,
            # "block_size": model_config.block_size,
            "vocab_size": model_config.vocab_size,
        }
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train GPT model on TinyStories (or another dataset).")
    parser.add_argument("--model-config", type=str, required=True, 
                        help="Which config key to load (e.g. gpt-1M, gpt-15M, etc.)")
    parser.add_argument("--ecoc-type", type=str, required=True, 
                        help="Which ecoc model to run.", default="minimal")
    parser.add_argument("--time-check", type=bool, required=True, 
                        help="Measure times for different steps.", default=False)
    return parser.parse_args()

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
    f"trainable params: {trainable_params} || "
    f"all params: {all_param} || "
    f"trainable%: {100 * trainable_params / all_param}"
)

def main():
    args = parse_args()
    if args.ecoc_type=="minimal qwen":
        model_config = get_qwen2_hf_config(config[args.model_config])
    else:
        model_config = config[args.model_config]
    logger.info("Using Model Config: %s", model_config)

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.time_check==False:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info("Using GPU for training.")
    elif args.time_check==True:
        device = 'cpu'
        logger.info("Time check mode: Using CPU for training.")
    logger.info("Using device: %s", device)

    train_data, val_data = load_data(
        config["data"],
        batch_size=model_config.batch_size,
        n=model_config.n,
        device=device
    )

    tokenizer = Tokenizer(
        config["tokenizer_qwen"],
        k=model_config.vocab_size - 1,
        file_path="tokens.json",
        device=device
    )

    model = get_model(args.ecoc_type, model_config, device, args.time_check)
             
    model = model.to(device)
    # for param in model.parameters():
    #     param.requires_grad = False  # Freeze all parameters

    # for param in model.ecoc_head.parameters():
    #     param.requires_grad = True  # Unfreeze only the LM head

    print_trainable_parameters(model)

    # optim = torch.optim.Adam(model.parameters(), lr=model_config.lr)
    optim = bnb.optim.PagedAdamW(model.parameters(), lr=model_config.lr)

    run_name = f"{config['wandb']['prefix']}-ecoc-{args.ecoc_type}-training-{args.model_config}-vocab-{model_config.vocab_size}-epochs-{model_config.epochs}"

    wandb_run = initialize_wandb(model_config, run_name)

    trainer = Trainer(
        model_config=model_config,
        model=model,
        time=args.time_check,
        optimizer=optim,
        train_data=train_data,
        val_data=val_data,
        encoder=tokenizer.encoder,
        device=device
    )

    checkpoint_path = trainer.train(checkpoint_path=f"{config['checkpoints']['location']}", run_name=run_name)

    logger.info(f"ECOC Training completed successfully. {checkpoint_path}")
    
    return checkpoint_path


if __name__ == "__main__":
    main()