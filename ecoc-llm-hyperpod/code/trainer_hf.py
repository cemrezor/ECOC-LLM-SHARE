import torch
import logging
import math
import os
import sys
import time
from evaluator import *

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, model_config, model, optimizer, train_data, val_data, encoder, scheduler=None, wandb_run=None, device="cuda", time):
        self.model_config = model_config
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_data = train_data
        self.val_data = val_data
        self.encoder = encoder
        self.device = device
        self.wandb_run = wandb_run
        self.time = time
        self.logger = logging.getLogger(__name__)

    def validate(self, step, eval_steps=50, top_k=5):
        self.model.eval()
        total_loss = 0
        steps = 0
        total_top_k_accuracy = 0
        total_batches = 0

        with torch.no_grad():
            for batch in self.val_data:
                
                #logger.info("[TMP] batch is : %s ", batch)
                
                if steps >= eval_steps:
                    break
                tokens = self.encoder(
                    batch, 
                    # max_length=self.model_config.block_size, 
                    padding="max_length", 
                    truncation=True
                ).to(self.device)
                
                #logger.info("[TMP] tokens is : %s ", tokens)

                # logits, aligned_targets, loss = self.model(tokens, tokens)
                output = self.model(input_ids=tokens, labels=tokens)
                logits = output.logits
                aligned_targets = output.aligned_targets
                loss = output.loss

                # logger.info("[TMP] logits is : %s ", logits)
                # logger.info(f"[TMP] logits.shape: {logits.shape}")
                
                # logger.info("[TMP] aligned_targets is : %s ", aligned_targets)     
                # logger.info(f"[TMP]  aligned_targets.shape: {aligned_targets.shape}")

                total_loss += loss.item()
                # logger.info(f"[TMP] loss value: {loss.item()}")  

                # decoded_tokens = self.model.decode_ecoc_predictions_from_logits(logits)
                # target_tokens = self.model.ecoc_to_token_ids_3d(aligned_targets)  
                
                # for i in range(logits.shape[0]):  # Show 3 sequences
                #     logger.info(f"[TMP] True Tokens:      {target_tokens[i].tolist()}")
                #     logger.info(f"[TMP] Predicted Tokens:{decoded_tokens[i].tolist()}")

                steps += 1                
                batch_top_k_accuracy = calculate_top_k_accuracy(logits, aligned_targets, self.model, k=top_k)
                total_top_k_accuracy += batch_top_k_accuracy
                total_batches += 1

                logger.info(f"Step {step}, Batch {steps}: Batch Top-{top_k} Accuracy: {batch_top_k_accuracy:.4f}")

        avg_loss = total_loss / steps
        perplexity = 1#calculate_perplexity(avg_loss)
        avg_top_k_accuracy = total_top_k_accuracy / total_batches

        logger.info(
            f"Step {step}: Validation Loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}, "
            f"Average Top-{top_k} Accuracy: {avg_top_k_accuracy:.4f}"
        )

        if self.wandb_run:
            self.wandb_run.log({
                "step": step,
                "val_loss": avg_loss,
                "perplexity": perplexity,
                f"avg_top_{top_k}_accuracy": avg_top_k_accuracy
            })

        return {"val_loss": avg_loss, "perplexity": perplexity, f"avg_top_{top_k}_accuracy": avg_top_k_accuracy}


    def train_one_epoch(self, epoch, eval_interval=200, eval_steps=50):
        eval_interval = self.model_config.eval_interval
        eval_steps = self.model_config.eval_steps
        
        self.model.train()
        total_loss = 0
        steps = 0

        for batch in self.train_data:
            tokens = self.encoder(
                batch,
                max_length = 256, 
                # max_length=self.model_config.block_size, 
                padding="max_length", 
                truncation=True
            ).to(self.device)
            output = self.model(input_ids=tokens, labels=tokens)
            loss = output.loss

            self.optimizer.zero_grad(set_to_none=True)
            
            # t = 5
            start_t5 = time.process_time()
            
            loss.backward()
            
            start_6 = time.process_time()
            if self.time==True:
                logger.info("Time taken between t=5 to t=6: %f", start_6 - start_t5)
                #self.logger.info("Time taken between t=5 to t=6", start_6 - start_t5)
            # print("Time taken between t=5 to t=6", start_6 - start_t5)
            # t = 6
            self.optimizer.step()
            if self.time==True:
                logger.info("Time taken between t=6 to t=7: %f", time.process_time() - start_6)
            # print("Time taken between t=6 to t=7", time.process_time() - start_t5)
            # t = 7

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

    def save_checkpoint(self, checkpoint_path, run_name, epoch):
        os.makedirs(checkpoint_path, exist_ok=True)
        model_save_path = f"{checkpoint_path}/{run_name}-epoch-{epoch}.bin"
        self.model.save(model_save_path)
        logger.info(f"Checkpoint saved at {model_save_path}")
        return model_save_path

    def train(self, checkpoint_path, run_name):
        last_checkpoint_path = None
        for epoch in range(1, self.model_config.epochs + 1):
            logger.info(f"Starting Epoch {epoch}/{self.model_config.epochs}")
            self.train_one_epoch(epoch)

            last_checkpoint_path = self.save_checkpoint(checkpoint_path, run_name, epoch)

        if self.wandb_run:
            self.wandb_run.finish()
        
        return last_checkpoint_path
