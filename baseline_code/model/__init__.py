from torch._C import Value
from .HCRN import HCRNNetwork
from .HCRN_bert import HCRNBertNetwork
from .HCRN_bert_freezed import HCRNFreezedBertNetwork


def get_model(model_name):
    if model_name == 'hcrn':
        return HCRNNetwork
    elif model_name == 'hcrn_bert':
        return HCRNBertNetwork
    elif model_name == 'hcrn_bert_freezed':
        return HCRNFreezedBertNetwork
    else:
        raise ValueError("No such model defined")