import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import transformers
from datasets import load_dataset
from huggingface_hub import login
from peft import LoraConfig, get_peft_model

print("=" * 80)
print("All imports completed successfully")
print("=" * 80)

model_id = "facebook/opt-1.3b"

try:
    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                 return_dict=True,
                                                 load_in_8bit=True,
                                                 device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Clear any cached memory to optimize GPU usage
    torch.cuda.empty_cache()

except Exception as e:
    print(f"Error loading model: {e}")


def prepare_model_for_finetuning(model):
    for param in model.parameters():
        param.requires_grad = False
        if param.ndim == 1:
            param.data = param.data.to(torch.float32)

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    class CastOutputToFloat(nn.Sequential):
        def forward(self, x): return super().forward(x).to(torch.float32)

    model.lm_head = CastOutputToFloat(model.lm_head)

prepare_model_for_finetuning(model)

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

config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
print("=" * 80)
print("Model loaded successfully")
model = get_peft_model(model, config)
print(model)
print_trainable_parameters(model)
print("=" * 80)

print("=" * 80)
data = load_dataset("Abirate/english_quotes")

data = data.map(lambda samples: tokenizer(samples['quote']), batched=True)
print("Successfully loaded the dataset")
print("=" * 80)

trainer = transformers.Trainer(
    model=model,
    train_dataset=data['train'],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=10,
        gradient_accumulation_steps=10,
        num_train_epochs=5,
        warmup_steps=300,
        max_steps=-1,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        output_dir='outputs'
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

trainer.train()
login(token="TOKEN")

model.push_to_hub("HF_PROFILE", use_auth_token=True)