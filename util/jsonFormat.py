'''
This module is a utils function for convertion between the json-style string and the model.state_dict()
Use for the information transmission between client and the blockchain.
The blockchain will receive a json-like modle in the smart contract.
'''
import json
from collections import OrderedDict
import string
try:
    import torch
except Exception:
    torch = None

def model2json(model_state_dict:OrderedDict):
    json_model = dict(model_state_dict)
    for name,value in json_model.items():
        json_model[name] = json_model[name].tolist()
    json_model = json.dumps(json_model)
    return json_model

def json2model(json_model_state_dict:string):
    model = json.loads(json_model_state_dict)
    if torch is None:
        raise RuntimeError("torch is required for json2model(); install torch before invoking this function")
    for name,value in model.items():
        model[name] = torch.Tensor(model[name])
    model = OrderedDict(model)
    return model
