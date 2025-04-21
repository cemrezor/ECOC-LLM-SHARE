from models.ecoc_min_model import MinimalEcocGPT2
from models.ecoc_ova_model import OvaECOCGPT2
from models.softmax_model import SoftmaxGPT2
from models.ecoc_ova_random_model import OvaPlusRandomECOCGPT2
#from models.ecoc_ova_2fc_model import OvaMTLECOCGPT2
#from models.ecoc_min_2fc_model import MinimalEcocMTLGPT2
from models.qwen_minimal_ecoc import Qwen2ForCausalLM

def get_model(model_type, model_config, device, time):
    if model_type == "minimal":
        model = MinimalEcocGPT2(model_config, device=device, time=time)
    elif model_type == "ova":
        model = OvaECOCGPT2(model_config, device=device, time=time)
    elif model_type == "softmax":
        model = SoftmaxGPT2(model_config, device=device, time=time)
    elif model_type == "ova_plus_random":
        model = OvaPlusRandomECOCGPT2(model_config, device=device, time=time)
    # elif model_type == "ova_MTL":
    #     model = OvaMTLECOCGPT2(model_config, device=device)
    # elif model_type == "min_MTL":
    #     model = MinimalEcocMTLGPT2(model_config, device=device)
    elif model_type == "minimal qwen":
        model = Qwen2ForCausalLM(model_config, device=device, time=time)
    else:
        raise ValueError(
            f"Invalid model type: {model_type}. "
            "Must be one of ['minimal','ova','softmax','ova_MTL','min_MTL','softmax qwen']"
        )
    return model