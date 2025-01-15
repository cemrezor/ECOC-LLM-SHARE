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
        "k": 7999,
        "vocab_size": 8000,
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
        "epochs": 3,
        "eval_interval": 200,
        "eval_steps": 50,
        "n": 1200000,
        "k": 1999,
        "vocab_size": 2000,
    },
     "tokenizer":{
        "name": "EleutherAI/gpt-neo-125M",
    },
    "data":{
        "name": "roneneldan/TinyStories",
    },

    "wandb" : {
        "project" : "Pretraining_ECOC_LLMs",
        "key" : "<>"
    }
}

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


config = Config(configdict)
