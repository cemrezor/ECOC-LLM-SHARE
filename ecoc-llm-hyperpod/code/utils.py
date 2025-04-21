import torch
from datasets import load_dataset, load_from_disk
import re
from torch.utils.data import DataLoader
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout
)

logger = logging.getLogger(__name__)

# def load_model(config, path, device='cpu'):
#     logger.info("Loading model from %s", path)
#     try:
#         model = GPT2(config, device=device)
#         model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
#         model.to(device)
#         model.eval()
#         logger.info("Model loaded successfully.", model)

#         return model
#     except Exception as e:
#         logger.error("Failed to load the model: %s", str(e))
#         raise e


def load_data(config, batch_size, n, device='cpu'):
    logger.info("Loading dataset...")
    try:
        if "data_location" in config:
            logger.info("Loading dataset from disk at %s", config["data_location"])
            dataset = load_from_disk(config["data_location"])
        else:
            logger.info("Downloading dataset from Hugging Face: %s", config["name"])
            dataset = load_dataset(config["name"])
    
        logger.info(f"dataset loaded to memory {dataset}. Preparing train and validation data loaders.")

        train_data = DataLoader(
            dataset["train"][:n]["text"], 
            batch_size=batch_size, 
            shuffle=True, 
            pin_memory=True, 
            pin_memory_device=device
        )
        val_data = DataLoader(
            dataset["validation"][:n]["text"], 
            batch_size=batch_size, 
            shuffle=True, 
            pin_memory=True, 
            pin_memory_device=device
        )
        logger.info("Dataset loaded successfully.")
        return train_data, val_data
    except Exception as e:
        logger.error("Failed to load dataset: %s", str(e))
        raise e

def clean_string(input_string):
    cleaned_string = re.sub(r'[^\w\s.,]', '', input_string)
    cleaned_string = re.sub(r'\s+', ' ', cleaned_string)
    cleaned_string = cleaned_string.replace('\n', '')
    return cleaned_string

@torch.no_grad()
def estimate_loss(model, train_data, val_data, encoder, eval_steps=50):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_steps)
        for k in range(eval_steps):
            data = train_data if split == 'train' else val_data
            tokens = encoder(next(iter(data))[0], max_length=model.block_size, padding="max_length", truncation=True)
            _, loss = model(tokens, tokens)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out