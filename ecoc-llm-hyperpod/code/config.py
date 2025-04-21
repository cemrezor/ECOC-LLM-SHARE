from transformers import Qwen2Config

configdict= {
    "gpt-1M":{
        "batch_size": 64,
        "block_size": 256,
        "max_pos_n_embed": 2048,
        "lr": 2e-3,
        "n_layer": 8,
        "n_head": 16,
        "n_embed": 64,
        "dropout": 0.2,
        "epochs": 1,
        "eval_interval": 200,
        "eval_steps": 50,
        "n": 1200000,
        "vocab_size": 8000, 
        "r": 40
    },
    "gpt-15M":{
        "batch_size": 64,
        "block_size": 256,
        "max_pos_n_embed": 2048,
        "lr": 2e-3,
        "n_layer": 8,
        "n_head": 16,
        "n_embed": 320,
        "dropout": 0.2,
        "epochs": 1,
        "eval_interval": 200,
        "eval_steps": 50,
        "n": 1200000,
        "vocab_size": 1000, 
        "r": 0
    },
    "gpt-30M": {
        "batch_size": 64,
        "block_size": 256,
        "max_pos_n_embed": 2048,
        "lr": 2e-3,
        "n_layer": 12,       
        "n_head": 16,        
        "n_embed": 384,      
        "dropout": 0.2,
        "epochs": 2,
        "eval_interval": 200,
        "eval_steps": 50,
        "n": 1200000,
        "vocab_size": 3000, 
        "r": 0
    },
    "qwen2": {
        "batch_size": 64,
        "lr": 2e-3,
        "dropout": 0.2,
        "epochs": 2,
        "eval_interval": 200,
        "eval_steps": 50,
        "n": 1200000,
        "attention_dropout": 0.0,
        "bos_token_id": 151643,
        "eos_token_id": 151643,
        "hidden_act": "silu",
        "hidden_size": 1536,
        "initializer_range": 0.02,
        "intermediate_size": 8960,
        "max_position_embeddings": 131072,
        "max_window_layers": 28,
        "model_type": "qwen2",
        "num_attention_heads": 12,
        "num_hidden_layers": 28,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-06,
        "rope_scaling": None,
        "rope_theta": 1000000.0,
        "sliding_window": None,
        "tie_word_embeddings": True,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.45.2",
        "use_cache": True,
        "use_sliding_window": False,
        "vocab_size": 8000
    },
    "tokenizer_qwen": {
        "name": "Qwen/Qwen2-1.5B"
    },
     "tokenizer":{
        "name": "EleutherAI/gpt-neo-125M",
    },
    "data":{
        "name": "roneneldan/TinyStories",
        "data_location" : "/fsx/ubuntu/ecoc-llm-env/data"
    },

    "wandb" : {
        "project" : "Pretraining_ECOC_LLMs",
        "key" : "d455a123a84819882762052288c93faa4531c2ab",
        "prefix" : "cz"
    },
    "checkpoints" : {
        "location" : "/fsx/ubuntu/ecoc-llm-env/checkpoints/czreproduce"
    }
}

def get_qwen2_hf_config():
    qcfg = configdict["qwen2"]
    return Qwen2Config(
        batch_size=qcfg["batch_size"],
        lr=qcfg["lr"],
        dropout=qcfg["dropout"],
        epochs=qcfg["epochs"],
        eval_interval=qcfg["eval_interval"],
        eval_steps=qcfg["eval_steps"],
        n=qcfg["n"],
        attention_dropout=qcfg["attention_dropout"],
        bos_token_id=qcfg["bos_token_id"],
        eos_token_id=qcfg["eos_token_id"],
        hidden_act=qcfg["hidden_act"],
        hidden_size=qcfg["hidden_size"],
        initializer_range=qcfg["initializer_range"],
        intermediate_size=qcfg["intermediate_size"],
        max_position_embeddings=qcfg["max_position_embeddings"],
        max_window_layers=qcfg["max_window_layers"],
        model_type=qcfg["model_type"],
        num_attention_heads=qcfg["num_attention_heads"],
        num_hidden_layers=qcfg["num_hidden_layers"],
        num_key_value_heads=qcfg["num_key_value_heads"],
        rms_norm_eps=qcfg["rms_norm_eps"],
        rope_scaling=qcfg["rope_scaling"],
        rope_theta=qcfg["rope_theta"],
        sliding_window=qcfg["sliding_window"],
        tie_word_embeddings=qcfg["tie_word_embeddings"],
        torch_dtype=qcfg["torch_dtype"],
        transformers_version=qcfg["transformers_version"],
        use_cache=qcfg["use_cache"],
        use_sliding_window=qcfg["use_sliding_window"],
        vocab_size=qcfg["vocab_size"]
    )

class Config:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def __getitem__(self, key):
        if key not in self.__dict__:
            raise KeyError(f"Key '{key}' not found in Config. Available keys: {list(self.__dict__.keys())}")
        return self.__dict__[key]

    def __iter__(self):
        return iter(self.__dict__.items())

    def size(self):
        return len(self.__dict__)
    
    def __contains__(self, key):
        return key in self.__dict__


config = Config(configdict)
