# ECOC-LLM
**ECOC-LLM** (Error-Correcting Output Code based Large Language Models) is a repository that provides tools and scripts for fine-tuning, evaluating, and inferencing large language models (LLMs) integrated with Error-Correcting Output Code (ECOC) technique. This repo focuses on enabling LLMs to handle complex output spaces by leveraging ECOC, which transforms multi-class classification into binary classification tasks, improving robustness and accuracy.

## Repository Structure

The repository is organized as follows:

- **Binary_code_dictionaries/**: Contains code dictionaries used for ECOC, which map vocabulary or labels into their respective binary code representations.
  
- **Evaluation_scripts/**: Scripts to evaluate the performance of the fine-tuned models, including precision, recall, f1 score.

- **Finetuning_scripts/**: Contains scripts for fine-tuning the LLM models with ECOC.

- **Inference/**: Notebooks for running inference with the fine-tuned models.

- **requirements.txt**: Lists the dependencies required to run the scripts in this repository.

## Getting Started

### Prerequisites

Before getting started, ensure you have the following installed:

- Python 3.10 or later
- PyTorch
- HuggingFace Transformers
- BitsAndBytes for 8-bit model loading
- Any other dependencies listed in `requirements.txt`

You can install the necessary dependencies by running:

```bash
pip install -r requirements.txt
```

### Fine-tuning

To fine-tune a model using ECOC, navigate to the `Finetuning_scripts/` directory and run the fine-tuning script:

```bash
cd Finetuning_scripts
python finetune.py
```

Ensure that the proper paths to datasets and model checkpoints are configured in the script.

### Evaluation

To evaluate a trained model, navigate to the `Evaluation_scripts/` directory and use the evaluation script:

```bash
cd Evaluation_scripts
python evaluate.py
```

You can adjust the evaluation parameters such as the test dataset, batch size, and evaluation metrics.

### Inference

For running inference, use the notebook provided in the `Inference/` directory. 

You can test the model on various input prompts and generate responses using the pre-trained or fine-tuned model.
