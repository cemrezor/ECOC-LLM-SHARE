import torch
import torch.nn as nn
from torch.nn import functional as F
import json
from typing import Callable, Optional, Tuple, Union, TypedDict
import numpy as np
import time
from models.gpt2_base import GPT2Base
from dataclasses import dataclass
from transformers import Qwen2Model, Qwen2PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.processing_utils import Unpack
import logging

@dataclass
class CausalLMOutputWithAlignedTargets(CausalLMOutputWithPast):
    """
    Extended causal LM output with an additional aligned_targets attribute.
    
    Args:
        aligned_targets (`torch.Tensor`, *optional*):
            Aligned targets for auxiliary supervision or metrics.
    """
    aligned_targets: Optional[torch.Tensor] = None

class LossKwargs(TypedDict, total=False):
    """
    Keyword arguments to be passed to the loss function

    Attributes:
        num_items_in_batch (`int`, *optional*):
            Number of items in the batch. It is recommended to pass it when
            you are doing gradient accumulation.
    """

    num_items_in_batch: Optional[int]

class FlashAttentionKwargs(TypedDict, total=False):
    """
    Keyword arguments for Flash Attention with Compile.

    Attributes:
        cu_seq_lens_q (`torch.LongTensor`, *optional*)
            Gets cumulative sequence length for query state.
        cu_seq_lens_k (`torch.LongTensor`, *optional*)
            Gets cumulative sequence length for key state.
        max_length_q (`int`, *optional*):
            Maximum sequence length for query state.
        max_length_k (`int`, *optional*):
            Maximum sequence length for key state.
    """

    cu_seq_lens_q: Optional[torch.LongTensor]
    cu_seq_lens_k: Optional[torch.LongTensor]
    max_length_q: Optional[int]
    max_length_k: Optional[int]

class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs):
    pass


class Qwen2ForCausalLM(Qwen2PreTrainedModel):
    # _tied_weights_keys = ["lm_head.weight"]
    # _tp_plan = {"lm_head": "colwise_rep"}
    # _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2Model(config)
        token_to_ecoc_map, ecoc_bits = self._generate_ecoc_codewords(config.vocab_size)
        self.ecoc_head = nn.Linear(config.hidden_size, ecoc_bits, bias=False)
        self.register_buffer(
            "ecoc_target_tensor",
            torch.tensor(
                [token_to_ecoc_map[token] for token in range(config.vocab_size)],
                dtype=torch.float32
            )
        )


        # Initialize weights and apply final processing
        # self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.ecoc_head

    def set_output_embeddings(self, new_embeddings):
        self.ecoc_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model
    
    def _generate_ecoc_codewords(self, vocab_size, r=0, seed=42):
        np.random.seed(seed)
        log2_v = int(np.ceil(np.log2(vocab_size))) 
        ecoc_bits = log2_v + r

        binary_codes = [format(i, f'0{log2_v}b') for i in range(vocab_size)]
        binary_matrix = np.array([[int(bit) for bit in code] for code in binary_codes])

        if r > 0:
            random_bits = np.random.randint(0, 2, (vocab_size, r))  
            binary_matrix = np.hstack((binary_matrix, random_bits))
        token_to_ecoc_map = {i: binary_matrix[i] for i in range(vocab_size)}

        return token_to_ecoc_map, ecoc_bits 


    # @can_return_tuple
    # @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    # @add_start_docstrings_to_model_forward(QWEN2_INPUTS_DOCSTRING)
    # @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> CausalLMOutputWithPast:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
                If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
                This is useful when using packed tensor format (single dimension for batch and sequence length).

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen2ForCausalLM

        >>> model = Qwen2ForCausalLM.from_pretrained("meta-qwen2/Qwen2-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-qwen2/Qwen2-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.ecoc_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            logits = logits[:, :-1, :]  # Trim the last token prediction
            shifted_targets = labels[:, 1:]
            aligned_targets = self.ecoc_target_tensor[shifted_targets.to(self.ecoc_target_tensor.device)].contiguous()
            # t = 2
            aligned_targets = aligned_targets.to(logits.device)
            start_t2 = time.process_time()
            # print(logits.shape, aligned_targets.shape)
            # print(logits.element_size() * logits.nelement() / 1e6, "MB")
            loss = F.binary_cross_entropy_with_logits(logits, aligned_targets.float())
            # loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithAlignedTargets(
            loss=loss,
            logits=logits,
            aligned_targets=aligned_targets,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def ecoc_logits_to_topk_tokens_3d(self, ecoc_logits, top_k=1):
        batch_size, sequence_length, ecoc_bits = ecoc_logits.shape

        probabilities = torch.sigmoid(ecoc_logits)
        
        probabilities_2d = probabilities.view(batch_size * sequence_length, ecoc_bits).float()
        
        target_tensor_float = self.ecoc_target_tensor.float()
        expanded_probs = probabilities_2d.unsqueeze(1)
        expanded_targets = target_tensor_float.unsqueeze(0)
        # expanded_targets = expanded_targets.to(expanded_probs.device)  # fix here

        diffs = (expanded_probs - expanded_targets) ** 2
        distances = diffs.sum(dim=-1)

        neg_distances = -distances
        top_k_indices = torch.topk(neg_distances, k=top_k, dim=1).indices

        top_k_tokens = top_k_indices.view(batch_size, sequence_length, top_k)

        return top_k_tokens

    def token_id_to_ecoc(self, token_id):
        return self.ecoc_target_tensor[token_id]

    def ecoc_to_token_ids_3d(self, targets):   
        batch_size, seq_length, _ = targets.shape
        tokens = torch.zeros((batch_size, seq_length), dtype=torch.long, device=targets.device)

        for i in range(batch_size):
            for j in range(seq_length):
                vec = targets[i, j]
                vec = vec.to(self.ecoc_target_tensor.device)
                exact_matches = (self.ecoc_target_tensor == vec).all(dim=1)
                if exact_matches.any():
                    token_id = torch.nonzero(exact_matches, as_tuple=True)[0].item()
                    tokens[i, j] = token_id
                else:
                    raise ValueError("Non existing ecoc code !!!")
        return tokens

    def generate(self, idx, max_tokens=20):
        """
        expecting 'idx' as shape (B, T).
        Typically, B=1 if you're doing single-sequence inference, but
        the model's forward pass is defined for (batch_size, seq_len).
        
        Returns shape => (B, T + max_tokens).
        """
        for _ in range(max_tokens):
            
            idx_cond = idx[:, -self.block_size:]  # shape => (B, <= block_size)
            logits, _, _ = self(idx_cond)
            last_logits = logits[:, -1:, :]
            
            top1 = self.ecoc_logits_to_topk_tokens_3d(last_logits, top_k=1)
            next_token = top1.squeeze(-1)
            idx = torch.cat((idx, next_token), dim=1)

        return idx

# @add_start_docstrings(
#     """
#     The Qwen2 Model transformer with a sequence classification head on top (linear layer).

#     [`Qwen2ForSequenceClassification`] uses the last token in order to do the classification, as other causal models
#     (e.g. GPT-2) do.

#     Since it does classification on the last token, it requires to know the position of the last token. If a
#     `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
#     no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
#     padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
#     each row of the batch).
#     """,
#     QWEN2_START_DOCSTRING,
# )

class MinimalEcocGPT2(GPT2Base):
  def __init__(self, config, device='cpu'):
        super().__init__(config, device=device)
        
        token_to_ecoc_map, ecoc_bits = self._generate_ecoc_codewords(config.vocab_size)
        
        self.ecoc_head = nn.Linear(config.n_embed, ecoc_bits)
        
        self.ecoc_target_tensor = torch.tensor(
            [token_to_ecoc_map[token] for token in range(config.vocab_size)], dtype=torch.float32
        ).to(self.device)
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"[Model] MinimalEcocGPT2 initialized with Ecoc bits: {ecoc_bits}")


  def _generate_ecoc_codewords(self, vocab_size, r=0, seed=42):
    np.random.seed(seed)
    log2_v = int(np.ceil(np.log2(vocab_size))) 
    ecoc_bits = log2_v + r

    binary_codes = [format(i, f'0{log2_v}b') for i in range(vocab_size)]
    binary_matrix = np.array([[int(bit) for bit in code] for code in binary_codes])

    if r > 0:
        random_bits = np.random.randint(0, 2, (vocab_size, r))  
        binary_matrix = np.hstack((binary_matrix, random_bits))
    token_to_ecoc_map = {i: binary_matrix[i] for i in range(vocab_size)}

    return token_to_ecoc_map, ecoc_bits 

  def forward(self, idx, targets=None):
    x = self.forward_gpt2_base(idx)  
    
    # # t = 0
    start_t0 = time.process_time()
    logits = self.ecoc_head(x)  # (B, T, ecoc_bits)
    # print(logits.shape)
    # print("Time taken between t=0 to t=1", time.process_time() - start_t0)
    # t = 1 

    if targets is None:
        aligned_targets = None
        loss = None
    else:
        logits = logits[:, :-1, :]
        # print("-1 operation", logits.shape)
        shifted_targets = targets[:, 1:]
        aligned_targets = self.ecoc_target_tensor[shifted_targets].contiguous()
        # t = 2
        start_t2 = time.process_time()
        loss = F.binary_cross_entropy_with_logits(logits, aligned_targets.float())
        # print("Time taken between t=2 to t=3", time.process_time() - start_t2)
        # t = 3

    return logits, aligned_targets, loss