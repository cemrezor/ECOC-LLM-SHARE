import torch
import random
import json
from transformers import AutoTokenizer
import math

def convert_token_id_to_binary_tensor_and_store(token_id, bit_size, bit_size_2):
    # Convert token_id to binary representation and pad with zeros to make it `bit_size` long
    binary_str = bin(token_id)[2:].zfill(bit_size)
    
    # Convert binary string to a tensor (or list for JSON serialization)
    binary_tensor = [int(bit) for bit in binary_str]
    
    # Add random bits to extend to `bit_size_2`
    if bit_size_2 > bit_size:
        extra_bits = [random.randint(0, 1) for _ in range(bit_size_2 - bit_size)]
        binary_tensor.extend(extra_bits)

    binary_string = ''.join(str(bit) for bit in binary_tensor)
    return binary_string, binary_tensor

def create_vocab_dictionary_from_tokenizer(tokenizer, bit_size, bit_size_2):
    vocab_dict = {}
    
    for token, token_id in tokenizer.vocab.items():
        binary_str, binary_tensor = convert_token_id_to_binary_tensor_and_store(token_id, bit_size, bit_size_2)
        vocab_dict[token_id] = binary_tensor   # Store binary tensor using integer token_id as key
        vocab_dict[binary_str] = token_id      # Store token_id using binary string as key
    
    return vocab_dict


model_id = "TinyLlama/TinyLlama_v1.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)  # Load a tokenizer, e.g., BERT's
bit_size = math.ceil(math.log2(len(tokenizer.vocab)))
bit_size_2 = 50
vocab_dict = create_vocab_dictionary_from_tokenizer(tokenizer, bit_size, bit_size_2)

# Export the dictionary to a JSON file
with open('vocab_dict_opt_50_tinyllama.json', 'w') as json_file:
    json.dump(vocab_dict, json_file)
