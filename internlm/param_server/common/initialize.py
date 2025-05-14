from collections import OrderedDict

import torch


def sorted_state_dict(unordered_dict):
    sorted_keys = sorted(unordered_dict.keys())  # 排序键名
    sorted_dict = OrderedDict()
    for key in sorted_keys:
        sorted_dict[key] = unordered_dict[key]
    return sorted_dict


def load_model(ckpt_path, num_layers, layer_chunk, param_shapes):
    """
    Load model from checkpoint file and split parameters into chunks. model should not be split.
        ckpt_path: str, path to the checkpoint file
        layer_chunk: list, list of layer indices for each chunk
        param_shapes: dict, dict of parameter shapes
    """
    state_dict = torch.load(ckpt_path, map_location="cpu")
    check_state_dict(state_dict, param_shapes)

    local_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("layers"):
            layer_idx = int(key.split(".")[1])
        else:
            if key.startswith("tok_embeddings") or key.startswith("embed_tokens"):
                layer_idx = 0
            elif key.startswith("norm") or key.startswith("output"):
                layer_idx = num_layers - 1

        if layer_idx in layer_chunk:
            # set to fp32
            local_state_dict[key] = value.float()

        value.requires_grad_()  # 设置梯度

    local_state_dict = sorted_state_dict(local_state_dict)

    return local_state_dict


def check_state_dict(state_dict, param_shapes):
    for key, value in state_dict.items():
        if key.startswith("layers"):
            key_split = key.split(".")
            param_key = ".".join(key_split[2:])
        else:
            param_key = key
        assert (
            value.shape == param_shapes[param_key]
        ), f"Shape mismatch for parameter {key}, expected {param_shapes[param_key]}, got {value.shape}"


OPTIMIZER_MAP = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
    "sgd_momentum": torch.optim.SGD,
    "nesterov": torch.optim.SGD,
}


def initialize_optimizer(opt_name, opt_kwargs, model_state_dict):
    params = model_state_dict.values()
    if opt_name == "nesterov":
        opt_kwargs["nesterov"] = True
    opt_kwargs.pop("name", None)
    optimizer = OPTIMIZER_MAP[opt_name](params, **opt_kwargs)
    return optimizer