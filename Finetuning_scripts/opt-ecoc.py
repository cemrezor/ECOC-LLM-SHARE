import os
import json
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, 
    OPTPreTrainedModel, 
    OPTModel, 
    Trainer,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
import transformers
from typing import Optional, Tuple, Union, List
from datasets import load_dataset
from huggingface_hub import login
from peft import LoraConfig, get_peft_model
import wandb

print("=" * 80)
print("All imports completed successfully")
print("=" * 80)

# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Initialize anomaly detection for debugging
torch.autograd.set_detect_anomaly(True)

# Configuration
MODEL_ID = "facebook/opt-1.3b"
VOCAB_DICT_PATH = ''
WANDB_API_KEY = ""
WANDB_PROJECT = ""
OUTPUT_DIR = ''
MODEL_NAME_HF = ''

# Initialize Weights & Biases
wandb.login(key=WANDB_API_KEY)
run = wandb.init(
    project=WANDB_PROJECT,
    config={
    "learning_rate": 2e-4,
    "architecture": "New OPT-ECoC with No LoRa on ECOC, LoRA on Attention and Alpaca Dataset with Masked BCELoss and Minimal ECOC, low LR, group_by_length as False",
    "dataset": "alpaca",
    "epochs": 3,
    }
)


class MaskedLoss(nn.Module):
    """
    Custom loss function that applies a mask to ignore certain target values.
    Uses Binary Cross Entropy Loss.
    """
    def __init__(self, ignore_value=-1.0):
        super(MaskedLoss, self).__init__()
        self.ignore_value = ignore_value
        self.cross_loss = nn.BCELoss(reduction='none')  # Compute element-wise loss

    def forward(self, input, target):
        # Create a mask to ignore entire rows filled with ignore_value
        mask = ~(target == self.ignore_value)
        input = input[mask]
        target = target[mask]
        loss = self.cross_loss(input, target)
        return loss.mean() if loss.numel() > 0 else torch.tensor(0.0, device=input.device)


def check_nan_in_weights(linear_layer):
    """
    Check if the weights of a linear layer contain NaN values.
    """
    weights = linear_layer.weight

    if weights.device.type == 'meta':
        print("Tensor is in Meta state. Cannot check for NaN values.")
        return False
    
    nan_mask = torch.isnan(weights)
    nan_indices = torch.nonzero(nan_mask, as_tuple=True)
    has_nan = nan_indices[0].numel() > 0

    if has_nan:
        print(f"NaN values found at indices: {nan_indices}")
    else:
        print("No NaN values found in the weights.")

    return has_nan


def check_nan_in_logits(logits):
    """
    Check if the logits contain NaN values.
    """
    if logits.device.type == 'meta':
        print("Tensor is in Meta state. Cannot check for NaN values.")
        return False

    nan_mask = torch.isnan(logits)
    nan_indices = torch.nonzero(nan_mask, as_tuple=True)
    has_nan = nan_indices[0].numel() > 0

    if has_nan:
        print(f"NaN values found at indices: {nan_indices}")
    else:
        print("No NaN values found in the logits.")

    return has_nan


def load_vocab_dictionary(file_path):
    """
    Load vocabulary dictionary from a JSON file.
    """
    with open(file_path, 'r') as json_file:
        vocab_dict = json.load(json_file)
    return vocab_dict


# Load vocabulary dictionary
token_binary_map = load_vocab_dictionary(VOCAB_DICT_PATH)


class OPTForCausalLM(OPTPreTrainedModel):
    """
    Custom OPT model for causal language modeling with ECOC head.
    """ 
    def __init__(self, config):
        super().__init__(config)
        self.model = OPTModel(config)

        # the lm_head weight is automatically tied to the embed tokens weight
        # self.lm_head = nn.Linear(config.word_embed_proj_dim, config.vocab_size, bias=False)
        self.bit_size = 16
        self.ecoc_head = nn.Linear(config.word_embed_proj_dim, self.bit_size, bias=False)
        self.ecoc_activation = nn.Sigmoid()
        self.post_init()

    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value

    def get_output_embeddings(self):
        return self.ecoc_head

    def set_output_embeddings(self, new_embeddings):
        self.ecoc_head = new_embeddings

    def set_decoder(self, decoder):
        self.model.decoder = decoder

    def get_decoder(self):
        return self.model.decoder
    
    def tie_weights(self):
        """
        Override to prevent weight tying.
        """
        pass

    def find_closest_tensor(self, given_tensor, tensor_of_tensors):
        """
        Find the closest tensor in tensor_of_tensors to the given_tensor.
        """
        assert given_tensor.shape[-1] == tensor_of_tensors.shape[-1], "Shape mismatch between given tensor and tensor of tensors"
        distances = torch.mean((tensor_of_tensors - given_tensor) ** 2, dim=tuple(range(1, tensor_of_tensors.dim())))
        min_index = torch.argmin(distances)
        min_dist, max_dist = distances.min(), distances.max()
        logits = 1 - (distances - min_dist) / (max_dist - min_dist + 1e-8)
        closest_tensor = tensor_of_tensors[min_index]
        min_distance = distances[min_index].item()
        return closest_tensor, min_distance, logits

    def int_to_bin_tensor(self, val):
        """
        Convert integer to binary tensor. Use -1 for ignore_value.
        """
        if val==-100:
            length = self.bit_size
            bin_str = format(2, '0' + str(length) + 'b')
            bin_tensor = torch.full((self.bit_size,), -1)
        else:
            length = self.bit_size
            bin_str = format(val, '0' + str(length) + 'b')
            bin_tensor = torch.tensor([int(bit) for bit in bin_str])
        return bin_tensor

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Get decoder outputs
        outputs = self.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # Compute ECOC logits
        logits = self.ecoc_activation(self.ecoc_head(outputs[0]).contiguous())

        loss = torch.tensor(0.0).to(logits.device)
        if labels is not None:
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            binary_tensors = []
            for i in range(shift_labels.shape[0]):
                binary_tensors_row = []
                for j in range(shift_labels.shape[1]):
                    val = shift_labels[i, j].item()
                    if val==-100:
                        binary_tensors_row.append(torch.full((self.bit_size,), -1))
                    else:
                        bin_tensor = self.int_to_bin_tensor(val)
                        binary_tensors_row.append(bin_tensor)
                binary_tensors.append(torch.stack(binary_tensors_row))
            
            binary_tensors = torch.stack(binary_tensors).to(logits.device)
            loss_fct = MaskedLoss()

            for j in range(logits.shape[-1]):  # Loop over each classifier
                # Compute loss for the j-th node in the final layer
                node_loss = loss_fct(shift_logits[:,:, j].float(), binary_tensors[:,:, j].float())
                # print(shift_logits[:,:, j].shape)
                loss += node_loss
            
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        """
        Prepare inputs for text generation.
        """
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past

try:
    model = OPTForCausalLM.from_pretrained(MODEL_ID,
                                                 return_dict=True,
                                                 load_in_8bit=False,
                                                 device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding=False)

    # Clear any cached memory to optimize GPU usage
    torch.cuda.empty_cache()

except Exception as e:
    print(f"Error loading model: {e}")


def prepare_model_for_finetuning(model):
    """
    Freeze all model parameters except those to be fine-tuned.
    Enable gradient checkpointing and input gradients.
    """
    for param in model.parameters():
        param.requires_grad = False
        if param.ndim == 1:
            param.data = param.data.to(torch.float32)

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

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
    
# Configure LoRA
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

# Apply LoRA to the model
model = get_peft_model(model, config)
print(model)

# Unfreeze ECOC head parameters for fine-tuning
for param in model.ecoc_head.parameters():
    param.requires_grad = True

# Unfreeze LoRA parameters
for name, param in model.named_parameters():
    if "lora_" in name:
        param.requires_grad = True

# Display which parameters are trainable
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"Parameter '{name}' is trainable (requires_grad=True).")

# Check for NaNs in weights
print(check_nan_in_weights(model.ecoc_head))

# Print trainable parameters summary
print_trainable_parameters(model)
print("=" * 80)

print("=" * 80)

# Load and preprocess dataset
data = load_dataset("tatsu-lab/alpaca")

def generate_prompt(data_point):
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{data_point["instruction"]}
### Input:
{data_point["input"]}
### Response:
{data_point["output"]}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
{data_point["instruction"]}
### Response:
{data_point["output"]}"""

# Shuffle and tokenize the dataset
data = data.shuffle().map(
    lambda data_point: tokenizer(
        generate_prompt(data_point),
        padding="longest",
        max_length=512,
        truncation=True,
    )
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    train_dataset=data['train'],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=50,
        gradient_accumulation_steps=1,
        num_train_epochs=3,
        max_grad_norm=1.0,
        warmup_ratio=0.1,
        max_steps=-1,
        learning_rate=2e-4,
        fp16=False,
        logging_steps=10,
        group_by_length = False,
        output_dir=OUTPUT_DIR,
        report_to=["wandb"],
        optim = "paged_adamw_32bit",
        save_steps=2000,
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

# Start training
trainer.train()

# Login to Hugging Face Hub
login(token="")

# Merge and unload LoRA adapters
trainer.model = trainer.model.merge_and_unload()
trainer.model.save_pretrained(OUTPUT_DIR)
trainer.model.push_to_hub(MODEL_NAME_HF, use_auth_token=True)