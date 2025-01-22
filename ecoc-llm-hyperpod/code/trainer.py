import torch
import logging
import math
import os
import sys

from evaluator import *

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, config, model, optimizer, train_data, val_data, encoder, scheduler=None, wandb_run=None, device="cuda"):
        self.config = config
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_data = train_data
        self.val_data = val_data
        self.encoder = encoder
        self.device = device
        self.wandb_run = wandb_run

    def validate(self, step, eval_steps=50, top_k=5):
        self.model.eval()
        total_loss = 0
        steps = 0
        top_k_correct = 0
        total_tokens = 0

        with torch.no_grad():
            for batch in self.val_data:
                if steps >= eval_steps:
                    break
                tokens = self.encoder(
                    batch, 
                    max_length=self.config.block_size, 
                    padding="max_length", 
                    truncation=True
                ).to(self.device)

                logits, loss = self.model(tokens, tokens)
                total_loss += loss.item()
                steps += 1

                top_k_correct += calculate_top_k_accuracy(logits, tokens, k=top_k)
                total_tokens += tokens.numel()

        avg_loss = total_loss / steps
        perplexity = calculate_perplexity(avg_loss)
        top_k_accuracy = top_k_correct / total_tokens

        logger.info(
            f"Step {step}: Validation Loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}, "
            f"Top-{top_k} Accuracy: {top_k_accuracy:.4f}"
        )

        if self.wandb_run:
            self.wandb_run.log({
                "step": step,
                "val_loss": avg_loss,
                "perplexity": perplexity,
                f"top_{top_k}_accuracy": top_k_accuracy
            })

        return {"val_loss": avg_loss, "perplexity": perplexity, f"top_{top_k}_accuracy": top_k_accuracy}

    

    def train_one_epoch(self, epoch, eval_interval=200, eval_steps=50):
        self.model.train()
        total_loss = 0
        steps = 0

        for batch in self.train_data:
            tokens = self.encoder(
                batch, 
                max_length=self.config.block_size, 
                padding="max_length", 
                truncation=True
            ).to(self.device)
        
            _, loss = self.model(tokens, tokens)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            if self.scheduler:
                self.scheduler.step()
            
            total_loss += loss.item()
            steps += 1
        
            if self.wandb_run:
                self.wandb_run.log({
                    "batch_train_loss": loss.item(), 
                    "step": steps
                })

            if steps % eval_interval == 0:
                logger.info(f"Performing validation at step {steps}")
                self.validate(steps, eval_steps=eval_steps)

        avg_loss = total_loss / steps
        logger.info(f"Epoch {epoch}: Train Loss: {avg_loss:.4f}")

        if steps % eval_interval != 0:
            logger.info(f"Performing end-of-epoch validation for epoch {epoch}")
            self.validate(steps, eval_steps=eval_steps)

        if self.wandb_run:
            self.wandb_run.log({
                "epoch_train_loss": avg_loss, 
                "epoch": epoch
            })

        return {"train_loss": avg_loss}

    def save_checkpoint(self, run_name):
        os.makedirs(self.config['checkpoints']['location'], exist_ok=True)
        checkpoint_path = f"{self.config['checkpoints']['location']}/{run_name}.bin"
        self.model.save(checkpoint_path)
        logger.info(f"Checkpoint saved at {checkpoint_path}")

    def train(self, eval_interval=200, eval_steps=50):
        for epoch in range(1, self.config.epochs + 1):
            logger.info(f"Starting Epoch {epoch}/{self.config.epochs}")
            self.train_one_epoch(epoch, eval_interval=eval_interval, eval_steps=eval_steps)

            run_name = f"training-{self.config['model_name']}-epoch-{epoch}"
            self.save_checkpoint(run_name)

        if self.wandb_run:
            self.wandb_run.finish()
