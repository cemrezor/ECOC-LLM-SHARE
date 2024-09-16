import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import torch.nn as nn
from transformers import AutoTokenizer, OPTPreTrainedModel, OPTModel
from transformers.modeling_outputs import CausalLMOutputWithPast
import transformers
from typing import Optional, Tuple, Union, List
from datasets import load_dataset
import math
import json
from bert_score import score as bert_score
from tqdm import tqdm

def load_vocab_dictionary(file_path):
    with open(file_path, 'r') as json_file:
        vocab_dict = json.load(json_file)
    return vocab_dict

token_binary_map = load_vocab_dictionary('/home/ec2-user/llms/ECOC/Training_scripts/vocab_dict_opt.json')

tensor_list = []
for value in token_binary_map.values():
    if isinstance(value, list):
        tensor = torch.tensor(value)
        tensor_list.append(tensor)
tensor_list = torch.stack(tensor_list)

class MaskedLoss(nn.Module):
    def __init__(self, ignore_value=-1.0):
        super(MaskedLoss, self).__init__()
        self.ignore_value = ignore_value
        self.cross_loss = nn.BCELoss(reduction='none')  # Compute element-wise loss

    def forward(self, input, target):
        # Create a mask to ignore entire rows filled with ignore_value
        mask = ~(target == self.ignore_value)
        # print(target)
        # Compute MSE loss
        input = input[mask]
        target = target[mask]
        loss = self.cross_loss(input, target)
        # nan_mask = torch.isnan(loss)
            
        # # Get the indices of NaN values
        # nan_indices = torch.nonzero(nan_mask, as_tuple=True)

        # print(nan_indices)
        # print("Input logits", input[nan_indices])
        # Apply the mask to ignore certain rows
        # print("before Loss ka shape", loss.shape)
        # print("Mask ka shape", mask.shape)
        # loss = loss[mask]
        # print("Loss after mask", loss)
        # print("after Loss ka shape", loss.shape)
        if loss.numel() > 0:
            loss = loss.mean()
        else:
            loss = torch.tensor(0.0, device=input.device)
        # # Calculate the mean of the remaining elements
        # loss = loss.mean()
        # print(loss)
        return loss

class OPTForCausalLM(OPTPreTrainedModel):
    # _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = OPTModel(config)

        # the lm_head weight is automatically tied to the embed tokens weight
        # self.lm_head = nn.Linear(config.word_embed_proj_dim, config.vocab_size, bias=False)
        self.bit_size = 50
        self.ecoc_head = nn.Linear(config.word_embed_proj_dim, self.bit_size, bias=False)
        self.ecoc_activation = nn.Sigmoid()
        # Initialize weights and apply final processing
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
        # Override this method to prevent weight tying
        pass

    def find_closest_tensor(self, given_tensor, tensor_of_tensors):
        assert given_tensor.shape[-1] == tensor_of_tensors.shape[-1], "Shape mismatch between given tensor and tensor of tensors"
        distances = torch.mean((tensor_of_tensors - given_tensor) ** 2, dim=tuple(range(1, tensor_of_tensors.dim())))
        min_index = torch.argmin(distances)
        min_dist, max_dist = distances.min(), distances.max()
        logits = 1 - (distances - min_dist) / (max_dist - min_dist + 1e-8)
        closest_tensor = tensor_of_tensors[min_index]
        min_distance = distances[min_index].item()
        return closest_tensor, min_distance, logits

    def int_to_bin_tensor(self, val):
        if val==-100:
            length = self.bit_size
            bin_str = format(2, '0' + str(length) + 'b')
            bin_tensor = torch.tensor([int(bit) for bit in bin_str])
        else:
            length = self.bit_size
            bin_str = format(val, '0' + str(length) + 'b')
            bin_tensor = torch.tensor([int(bit) for bit in bin_str])
        return bin_tensor

    # @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
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
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(num_hidden_layers, num_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
                shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`. The two additional
                tensors are only required when the model is used as a decoder in a Sequence to Sequence model.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, OPTForCausalLM

        >>> model = OPTForCausalLM.from_pretrained("facebook/opt-350m")
        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious. I'm just a little bit of a weirdo."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
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

        logits = self.ecoc_activation(self.ecoc_head(outputs[0]).contiguous())

        loss = torch.tensor(0.0).to(logits.device)
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            binary_tensors = []
            for i in range(shift_labels.shape[0]):
                binary_tensors_row = []
                for j in range(shift_labels.shape[1]):
                    val = shift_labels[i, j].item()
                    # print(val)
                    if val==-100:
                        binary_tensors_row.append(torch.full((self.bit_size,), -1))
                    else:
                        bin_tensor = torch.tensor(token_binary_map[str(val)])
                        binary_tensors_row.append(bin_tensor)
                binary_tensors.append(torch.stack(binary_tensors_row))
            binary_tensors = torch.stack(binary_tensors)
            binary_tensors = binary_tensors.to(logits.device)
            loss_fct = MaskedLoss()
            # print(shift_logits.shape)
            # print(binary_tensors.shape)
            # total_loss = torch.tensor(0.0).to(logits.device)
            # print("New for loop instantiate")
            for j in range(logits.shape[-1]):  # Loop over each classifier
                # Compute loss for the j-th node in the final layer
                node_loss = loss_fct(shift_logits[:,:, j].float(), binary_tensors[:,:, j].float())
                # print(shift_logits[:,:, j].shape)
                loss += node_loss
            #     loss_array.append(node_loss)
            #     node_loss.backward(retain_graph=True)
            # for param in model.ecoc_head.parameters():
            #     if param.grad is not None:
            #         Individual = param.grad
            #         print(param.grad)
            #         param.grad.zero_()
            #         print(param.grad)
            # print("Loss summation")
            # total_loss.backward(retain_graph=True)
            # for param in model.ecoc_head.parameters():
            #     if param.grad is not None:
            #         Summation = param.grad
            #         print(param.grad)
            #         param.grad.zero_() 
            #     # self.accelerator.backward(node_loss, retain_graph=True)
            # print("Equal or not", torch.equal(Individual, Summation))
            # loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            # print(loss)
            # check_nan_in_logits(logits)
            # check_nan_in_weights(self.ecoc_head)
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
    
model = OPTForCausalLM.from_pretrained("/home/ec2-user/llms/ECOC/Training_scripts/new_opt_finetune_ecoc_Random_alpaca_with_lora_groupFalse_low_LR_BCE", return_dict=True, load_in_8bit=False, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")

print(model)

data = load_dataset("disham993/alpaca-train-validation-test-split")

def generate_prompt(data_point):
    if data_point["input"]:
        return f""" ### Instruction:
{data_point["instruction"]}
### Input:
{data_point["input"]}
### Response:
"""
    else:
        return f""" ### Instruction:
{data_point["instruction"]}
### Response:
"""
def int_to_bin_tensor(val, bit_size):
    if val==-100:
        length = bit_size
        bin_tensor = torch.full((bit_size,), -1)
    else:
        length = bit_size
        bin_str = format(val, '0' + str(length) + 'b')
        bin_tensor = torch.tensor([int(bit) for bit in bin_str])
    return bin_tensor
    
def bin_tensor_to_int(bin_tensor):
    """Convert a binary tensor to an integer."""
    bin_str = ''.join(str(bit.item()) for bit in bin_tensor)
    return int(bin_str, 2)

def convert_vocabulary_to_binary(tokenizer):
    """Convert all token IDs in the tokenizer's vocabulary to binary with specified bit size."""
    # Get the vocabulary
    vocab = tokenizer.get_vocab()
    bit_size = math.ceil(math.log2(tokenizer.vocab_size))
    
    # Convert each token ID to binary with the specified bit size
    binary_vocab = [int_to_bin_tensor(token_id, bit_size) for token_id in vocab.values()]

    # Stack the individual binary code tensors into a single tensor of tensors
    binary_vocab_tensor = torch.stack(binary_vocab)

    return binary_vocab_tensor

binary_vocab = convert_vocabulary_to_binary(tokenizer)

max_length = 30
def generate_predictions(data_split):
    predictions = []
    for data_point in tqdm(data_split, desc="Generating Predictions"):
        generated_ids = []
        prompt = generate_prompt(data_point)
        inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
        inputs = inputs['input_ids']
        # attention_mask = inputs['attention_mask']
        # # If you need to combine them into a single tensor (e.g., stacking)
        # inputs = torch.cat((input_ids, attention_mask), dim=0)
        for _ in range(max_length - inputs.size(1)):
            model_inputs = model.prepare_inputs_for_generation(inputs)
            with torch.no_grad():
                outputs = model(**model_inputs, return_dict=True)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = model.find_closest_tensor(next_token_logits.to(next_token_logits.device), tensor_list.to(next_token_logits.device))
            binary_string = ''.join(str(int(bit)) for bit in next_token[0])
            next_token_id = torch.tensor(token_binary_map[binary_string]).to(model.device)
            # next_token_id = torch.tensor(bin_tensor_to_int(next_token[0])).to(model.device)
            generated_ids.append(next_token_id.unsqueeze(0).unsqueeze(0))
            # Stop generation if end-of-sequence token is generated (optional)
            if next_token_id == tokenizer.eos_token_id:
                break
        decoded_output = tokenizer.decode(torch.tensor(generated_ids), skip_special_tokens=True)
        predictions.append(decoded_output.strip())
    return predictions


validation_predictions = generate_predictions(data["test"].select(range(1000)))
validation_references = [data_point["output"] for data_point in data["test"].select(range(1000))]

# Function to calculate BERTScore
def calculate_bertscore(predictions, references):
    precision, recall, f1 = bert_score(predictions, references, lang="en", verbose=True)
    return precision.mean().item(), recall.mean().item(), f1.mean().item()

# Calculate BERTScore for test sets
validation_precision, validation_recall, validation_f1 = calculate_bertscore(validation_predictions, validation_references)

print(f"Validation BERTScore for Minimal ECOC OPT - Precision: {validation_precision:.4f}, Recall: {validation_recall:.4f}, F1: {validation_f1:.4f}")
