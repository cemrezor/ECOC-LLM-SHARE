import argparse
import torch
from tokenizer import Tokenizer
from config import config
from models.model_factory import get_model
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")

def parse_args():
    parser = argparse.ArgumentParser(description="Inference script for GPT models (ECOC or softmax).")
    parser.add_argument("--model-config", type=str, required=True, 
                        help="Which config key to load (e.g. gpt-1M, gpt-15M, etc.)")
    parser.add_argument("--model-type", type=str, required=True, 
                        help="Which model variant to run. (e.g. minimal, ova, softmax, ova_MTL, min_MTL, softmax qwen)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the .bin checkpoint file containing model state.")
    parser.add_argument("--prompt", type=str, default="Hello world",
                        help="Initial text to generate from.")
    parser.add_argument("--max_tokens", type=int, default=20,
                        help="Number of new tokens to generate.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"],
                        help="Inference device.")
    return parser.parse_args()

def generate_text(args):
    logger.info(f"Loading model config: {args.model_config}")
    model_config = config[args.model_config]

    logger.info(f"Creating model of type {args.model_type}")
    model = get_model(args.model_type, model_config, device=args.device)

    logger.info(f"Loading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(args.device)
    model.eval()

    vocab_size = model_config["vocab_size"]
    
    tokenizer = Tokenizer(
        config["tokenizer"],  
        k=vocab_size - 1,
        file_path="tokens.json",
        device=args.device
    )

    logger.info(f"Encoding prompt: '{args.prompt}'")
    encoded_prompt = tokenizer.encoder([args.prompt], padding=False, truncation=False)

    with torch.no_grad():
        logger.info(f"Generating up to {args.max_tokens} tokens...")
        generated_tokens = model.generate(
            idx=encoded_prompt,
            max_tokens=args.max_tokens
        )
    
    text = tokenizer.decoder(generated_tokens)
    return text

if __name__ == "__main__":

    args = parse_args()
    generated_text = generate_text(args)

    logger.info("===== Generated Text =====")
    print(generated_text)
