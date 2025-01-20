from utils import *

class Trainer:
    def __init__(self, config, model, optimizer, train_data, val_data, encoder, wandb_run=None):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.train_data = train_data
        self.val_data = val_data
        self.encoder = encoder
        self.wandb_run = wandb_run

        if self.wandb_run:
            self.wandb_run.config.update({
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "learning_rate": config.lr,
                "block_size": config.block_size,
                "vocab_size": config.vocab_size,
            })


    def train(self, epochs, eval_interval=200, eval_steps=50):
        max_steps = epochs * round(self.config.n / self.config.batch_size)
        steps = 0
        tracked_losses = []

        for epoch in range(epochs):
            print(f"Starting Epoch: {epoch + 1} {'-' * 100}")
            for batch in self.train_data:
                # Evaluate and log losses at intervals
                if steps % eval_interval == 0 or steps == max_steps - 1:
                    losses = estimate_loss(self.model, self.train_data, self.val_data, self.encoder, eval_steps)
                    tracked_losses.append(losses)

                    # Log losses to WandB
                    self.wandb_run.log({
                        "epoch": epoch + 1,
                        "step": steps,
                        "train_loss": losses['train'],
                        "val_loss": losses['val']
                    })

                    print(
                        f"Epoch: {epoch + 1}/{epochs} | Step: {steps}/{max_steps} "
                        f"| Train loss: {losses['train']:.4f} | Val loss: {losses['val']:.4f}"
                    )

                # Process batch and perform optimization step
                tokens = self.encoder(
                    batch, 
                    max_length=self.config.block_size, 
                    padding="max_length", 
                    truncation=True
                )
                _, loss = self.model(tokens, tokens)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

                # Log training loss to WandB for each step
                self.wandb_run.log({
                    "step": steps,
                    "batch_loss": loss.item()
                })

                steps += 1

        return tracked_losses

    def finalize(self):
        """Call this method to finalize the WandB run."""
        if self.wandb_run:
            self.wandb_run.finish()
