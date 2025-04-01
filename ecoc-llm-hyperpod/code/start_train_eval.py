#!/usr/bin/env python3
import sys
import start_training
import eval

def main():

    checkpoint_path = start_training.main()
    sys.argv = [
        "eval.py",                               # pretend script name
        "--checkpoint_path", f"{checkpoint_path}",
        "--data_path",      "/fsx/ubuntu/ecoc-llm-env/data/validation",
        "--device",         "cuda",
        "--use_wandb",      # if you want W&B logging
    ]
    eval.main()

    print("All done — training and evaluation completed successfully.")

if __name__ == "__main__":
    main()
