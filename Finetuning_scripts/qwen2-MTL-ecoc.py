import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
)
from transformers.cache_utils import StaticCache
import transformers
from peft import LoraConfig, get_peft_model
from typing import Optional, Tuple, Union, List, Any
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import replace_return_docstrings
from transformers import Qwen2PreTrainedModel, Qwen2Model, AutoTokenizer, DataCollatorForLanguageModeling, Trainer
from transformers.file_utils import add_start_docstrings_to_model_forward
import torch.nn as nn
import math, wandb, json
from huggingface_hub import login

print("=" * 80)
print("All imports completed successfully")
print("=" * 80)

# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Enable anomaly detection for debugging
torch.autograd.set_detect_anomaly(True)

MODEL_ID = "Qwen/Qwen2-1.5B"
VOCAB_DICT_PATH = ''
WANDB_API_KEY = ""
WANDB_PROJECT = ""
OUTPUT_DIR = ''
MODEL_NAME_HF = ''
_CONFIG_FOR_DOC = "Qwen2Config"
wandb.login(key=WANDB_API_KEY)
run = wandb.init(
    # set the wandb project where this run will be logged
    project=WANDB_PROJECT,
    config={
    "learning_rate": 2e-4,
    "architecture": "New MTL Qwen2-ECoC with No LoRa on ECOC, LoRA on Attention and Alpaca Dataset with Masked BCELoss and Random ECOC 50 bits, low LR, group_by_length as False, 512 hidden state, 3 epochs",
    "dataset": "alpaca",
    "epochs": 3,
    }
)
# Custom Loss Function
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

def _prepare_4d_causal_attention_mask_with_cache_position(
    attention_mask: torch.Tensor,
    sequence_length: int,
    target_length: int,
    dtype: torch.dtype,
    device: torch.device,
    min_dtype: float,
    cache_position: torch.Tensor,
    batch_size: int,
):
    """
    Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.
    Args:
        attention_mask (`torch.Tensor`):
            A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
        sequence_length (`int`):
            The sequence length being processed.
        target_length (`int`):
            The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
        dtype (`torch.dtype`):
            The dtype to use for the 4D attention mask.
        device (`torch.device`):
            The device to plcae the 4D attention mask on.
        min_dtype (`float`):
            The minimum value representable with the dtype `dtype`.
        cache_position (`torch.Tensor`):
            Indices depicting the position of the input sequence tokens in the sequence.
        batch_size (`torch.Tensor`):
            Batch size.
    """
    if attention_mask is not None and attention_mask.dim() == 4:
        # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
        causal_mask = attention_mask
    else:
        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_dtype
            )

    return causal_mask


QWEN2_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance;
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""

class Qwen2ForCausalLM(Qwen2PreTrainedModel):
    """
    Custom Qwen2 model for causal language modeling with MTL-ECOC head.
    """
    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2Model(config)
        self.vocab_size = config.vocab_size
        # self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # self.bit_size = math.ceil(math.log2(self.vocab_size))
        self.bit_size =  50
        self.linear_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_size, 512, bias=False),  # First linear layer
                nn.ReLU(),  # Activation function
                nn.Linear(512, 1, bias=False),  # Second linear layer, outputting a single logit
                nn.Sigmoid()  # Sigmoid activation to convert logits to probabilities (0 or 1)
            )
            for _ in range(self.bit_size)
        ])

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.linear_heads

    def set_output_embeddings(self, new_embeddings):
        self.linear_heads = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

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
            bin_tensor = torch.full((self.bit_size,), -1)
        else:
            length = self.bit_size
            bin_str = format(val, '0' + str(length) + 'b')
            bin_tensor = torch.tensor([int(bit) for bit in bin_str])
        return bin_tensor

    @add_start_docstrings_to_model_forward(QWEN2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Get decoder outputs
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        # Compute MTL-ECOC logits
        logits = torch.stack([head(hidden_states) for head in self.linear_heads])
        logits = logits.squeeze(-1)  
        logits = logits.permute(1, 2, 0)  # Shape: [batch_size, seq_len, bit_size]

        # Compute loss if labels are provided
        loss = torch.tensor(0.0).to(logits.device)
        if labels is not None:
            labels = labels.to(logits.device)
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
                        bin_tensor = torch.tensor(token_binary_map[str(val)])
                        binary_tensors_row.append(bin_tensor)
                binary_tensors.append(torch.stack(binary_tensors_row))
            binary_tensors = torch.stack(binary_tensors)
            binary_tensors = binary_tensors.to(logits.device)
            loss_fct = MaskedLoss()

            for j in range(logits.shape[-1]):  # Loop over each classifier
                # Compute loss for the j-th node in the final layer
                node_loss = loss_fct(shift_logits[:,:, j].float(), binary_tensors[:,:, j].float())
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
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        **kwargs,
    ):
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

                # This `clone` call is needed to avoid recapturing cuda graphs with `torch.compile`'s  `mode="reduce-overhead`, as otherwise the input `position_ids` would have various stride during the decoding. Here, simply using `.contiguous()` is not sufficient as in the batch size = 1 case, `position_ids` is already contiguous but with varying stride which retriggers a capture.
                position_ids = position_ids.clone(memory_format=torch.contiguous_format)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
            if inputs_embeds is not None:
                batch_size, sequence_length = inputs_embeds.shape
                device = inputs_embeds.device
            else:
                batch_size, sequence_length = input_ids.shape
                device = input_ids.device

            dtype = self.lm_head.weight.dtype
            min_dtype = torch.finfo(dtype).min

            attention_mask = _prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=sequence_length,
                target_length=past_key_values.get_max_length(),
                dtype=dtype,
                device=device,
                min_dtype=min_dtype,
                cache_position=cache_position,
                batch_size=batch_size,
            )

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

try:
    model = Qwen2ForCausalLM.from_pretrained(MODEL_ID,
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

# Unfreeze MTL-ECOC head parameters for fine-tuning
for param in model.linear_heads.parameters():
    param.requires_grad = True

# Unfreeze parameters where LoRA is applied (they are already set up by LoRA)
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

# Load the dataset
data = load_dataset("disham993/alpaca-train-validation-test-split")

# Merge the 'train' and 'validation' splits into one 'train' split
data['train'] = data['train'].concatenate(data['validation'])

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
        per_device_train_batch_size=40,
        gradient_accumulation_steps=1,
        num_train_epochs=2,
        max_grad_norm=1.0,
        warmup_ratio=0.1,
        max_steps=-1,
        learning_rate=2e-4,
        fp16=False,
        logging_steps=10,
        group_by_length = False,
        output_dir=OUTPUT_DIR,
        report_to=["wandb"],
        # lr_scheduler_type = "cosine_with_restarts",
        optim = "paged_adamw_32bit",
        save_steps=2000,
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

# Disable caching during training
model.config.use_cache = False

# Start training
trainer.train()

# Login to Hugging Face Hub
login(token="")

# Merge and unload LoRA adapters
trainer.model = trainer.model.merge_and_unload()
trainer.model.save_pretrained(OUTPUT_DIR)
trainer.model.push_to_hub(MODEL_NAME_HF, use_auth_token=True)