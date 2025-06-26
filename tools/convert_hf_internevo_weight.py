import os
import argparse
import torch
from tqdm import tqdm
from safetensors.torch import load_file

hf_internevo_key_mapping = {}


hf_internevo_key_mapping["QWEN3_30B_A3B"] = {
    "k_norm.weight": "attention.k_norm.weight",
    "k_proj.weight": "attention.wk.weight",
    "q_proj.weight": "attention.wq.weight",
    "v_proj.weight": "attention.wv.weight",
    "o_proj.weight": "attention.wo.weight",
    "q_norm.weight": "attention.q_norm.weight",
    "gate.weight": "feed_forward.moe_layer.gate.wg.weight",
    "post_attention_layernorm.weight": "ffn_norm.weight",
    "gate_proj.weight": "w1.weight",
    "up_proj.weight": "w3.weight",
    "down_proj.weight": "w2.weight",
    "lm_head.weight": "output.weight",
    "input_layernorm.weight": "attention_norm.weight",
    "embed_tokens.weight": "embed_tokens.weight",
    "norm.weight": "norm.weight"
}

hf_internevo_key_mapping["QWEN3_32B"] = {
    "k_norm.weight": "attention.k_norm.weight",
    "k_proj.weight": "attention.wk.weight",
    "q_proj.weight": "attention.wq.weight",
    "v_proj.weight": "attention.wv.weight",
    "o_proj.weight": "attention.wo.weight",
    "q_norm.weight": "attention.q_norm.weight",
    "post_attention_layernorm.weight": "ffn_norm.weight",
    "gate_proj.weight": "feed_forward.w1.weight",
    "up_proj.weight": "feed_forward.w3.weight",
    "down_proj.weight": "feed_forward.w2.weight",
    "lm_head.weight": "output.weight",
    "input_layernorm.weight": "attention_norm.weight",
    "embed_tokens.weight": "embed_tokens.weight",
    "norm.weight": "norm.weight"
}

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("origin_weight_path", type=str, default=None)
    args.add_argument("target_weight_path", type=str, default=None)
    args.add_argument("--model_type", type=str, default="QWEN3_A30_3B")
    return args.parse_args()

def load_all_safetensors(path):
    state_dict = {}
    for fname in tqdm(sorted(os.listdir(path))):
        if fname.endswith(".safetensors"):
            full_path = os.path.join(path, fname)
            tensors = load_file(full_path)
            state_dict.update(tensors)
    return state_dict

def map_keys(state_dict, model_type):
    # only support qwen3 a30 3b now
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            key = key[6:]
        new_key = key
        if "layers" in key:
            parts = key.split('.')
            layer_id = int(parts[1])
            suffix = '.'.join(parts[-2:])
            if "experts" in key:
                expert_id = int(parts[4])
                suffix = hf_internevo_key_mapping[model_type][suffix]
                new_key = '.'.join(parts[:2]) + ".feed_forward.moe_layer.experts.wrapped_experts." + str(expert_id) + "." + suffix
            else:
                suffix = hf_internevo_key_mapping[model_type][suffix]
                new_key = '.'.join(parts[:2]) + "." + suffix
        else:
            new_key = hf_internevo_key_mapping[model_type][key]

        print("new_key : ", new_key)
        new_state_dict[new_key] = value

    return new_state_dict

if __name__ == "__main__":
    args = parse_args()
    print(f"Loading hf weight from path {args.origin_weight_path}")
    all_weights = load_all_safetensors(args.origin_weight_path)
    print(f"Mapping hf key to internevo key")
    result = map_keys(all_weights, args.model_type)
    print(f"Saving result to {args.target_weight_path}")
    torch.save(result, args.target_weight_path)
    print("ok")
