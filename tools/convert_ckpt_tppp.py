# Copyright (c) InternLM. All rights reserved.
"""
Usage: python -W ignore convert_model_parallel.py \
    <origin_model_path> <target_model_path> \
    [--target_tp_size <target_tp_size>] [--target_pp_size <target_pp_size>] [--model_size <model_size>]

Example:
    srun -p llm_o --gres=gpu:8 python -W ignore tools/convert_model_parallel.py \
        /mnt/inspurfs/models/Gauss_20B/52000 /mnt/inspurfs/models/Gauss_20B/52000_converted \
        --target_tp_size 4 --target_pp_size 4 --model_size 20B
"""
import argparse
import os
import shutil
import sys
from collections import defaultdict

import torch

sys.path.append("./internlm/")
try:
    from internlm.utils.storage_manager import (
        get_fns,
        init_storage_manager,
        llm_load,
        llm_save,
    )

    init_storage_manager(False, None, None)
except ImportError:

    def get_fns(folder):
        return os.listdir(folder)

    def llm_load(fn, map_location="cpu"):
        return torch.load(fn, map_location=map_location)

    def llm_save(fp, obj):
        return torch.save(obj, fp)


using_qwen = True


PIPELINE_SHARDING_INFO = {
    "7B": {
        "1": [[(0, 32)]],
        "2": [[(0, 16)], [(16, 32)]],
        "4": [[(0, 8)], [(8, 16)], [(16, 24)], [(24, 32)]],
    },
    "20B": {
        "1": [[(0, 48)]],
        "2": [[(0, 24)], [(24, 48)]],
        "4": [[(0, 12)], [(12, 24)], [(24, 36)], [(36, 48)]],
    },
    "70B": {
        "1": [[(0, 80)]],
        "2": [[(0, 40)], [(40, 80)]],
        "4": [[(0, 20)], [(20, 40)], [(40, 60)], [(60, 80)]],
        "8": [[(0, 10)], [(10, 20)], [(20, 30)], [(30, 40)], [(40, 50)], [(50, 60)], [(60, 70)], [(70, 80)]],
    },
}

PIPELINE_SHARDING_INFO_QWEN = {
    "7B": {
        "1": [[(0, 28)]],
        "2": [[(0, 14)], [(14, 28)]],
    },
}

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("origin_model_path", type=str, default=None)
    args.add_argument("target_model_path", type=str, default=None)
    args.add_argument("--target_tp_size", type=int, default=0)
    args.add_argument("--target_pp_size", type=int, default=0)
    args.add_argument("--model_size", type=str, default="7B", choices=["7B", "20B", "70B"])
    return args.parse_args()


def get_mapping(n, m):
    if m % n != 0:
        raise ValueError("m must be a multiple of n")

    n_list = list(range(n))
    m_list = list(range(m))

    mapping = {}
    for i, n_val in enumerate(n_list):
        mapping[n_val] = m_list[i * int(m / n) : (i + 1) * int(m / n)]

    return mapping


def get_parallel_infos(folder):
    """Get the model parallel size and the pipeline parallel size of a model."""
    assert folder is not None, "Please specify the folder of the model to be converted."
    fns = get_fns(folder)

    model_fns = []
    for fn in fns:
        # filter with `_t` is for avoiding conflict with model_config.py
        if fn.startswith("model_t") and not fn.endswith("md5"):
            model_fns.append(fn)

    tp_size, pp_size = -1, -1
    for fn in model_fns:
        _, tp, pp = os.path.splitext(fn)[0].split("_")
        pp_size = max(pp_size, int(pp[2:]) + 1)
        tp_size = max(tp_size, int(tp[2:]) + 1)

    return pp_size, tp_size, sorted(model_fns)


def convert_tp_size(folder, saved_folder, target_tp_size):
    """Convert the model parallel size of a model."""
    pp_size, tp_size, _ = get_parallel_infos(folder)

    if tp_size > target_tp_size:
        assert tp_size % target_tp_size == 0, f"Cannot convert {tp_size} TP to {target_tp_size} TP."
    elif tp_size < target_tp_size:
        split_maps = get_mapping(tp_size, target_tp_size)
        assert target_tp_size % tp_size == 0, f"Cannot convert {tp_size} TP to {target_tp_size} TP."
    else:
        if os.path.exists(saved_folder):
            shutil.rmtree(saved_folder)
        shutil.copytree(folder, saved_folder)
        print(f"Model has already been converted to {target_tp_size} TP. Copied to {saved_folder}.")
        return

    # rules for tensor parallel splitting
    raw_dim, column_dim = 0, 1
    ratio = tp_size // target_tp_size if tp_size > target_tp_size else target_tp_size // tp_size

    def split(split_maps):
        for pp in range(pp_size):
            new_tp_states = [{} for _ in range(target_tp_size)]
            for tp in range(tp_size):
                states_tp = llm_load(os.path.join(folder, f"model_tp{tp}_pp{pp}.pt"))
                for k, v in states_tp.items():
                    assert len(v.size()) < 3, "Only support 2D or 1D tensors."
                    if len(v.size()) == 1:
                        assert "norm" in k
                        for i, map_id in enumerate(split_maps[tp]):
                            new_tp_states[map_id][k] = v
                    else:
                        if k == "tok_embeddings.weight" or "wo" in k or "w2" in k:
                            column_size = v.size()[column_dim] // ratio
                            new_tp_splits = torch.split(v, column_size, dim=column_dim)

                        elif k == "output.weight" or "w1" in k or "w3" in k or "wqkv" in k:
                            raw_size = v.size()[raw_dim] // ratio
                            new_tp_splits = torch.split(v, raw_size, dim=raw_dim)
                        else:
                            raise KeyError(f"Unknown key {k}.")

                        for i, map_id in enumerate(split_maps[tp]):
                            new_tp_states[map_id][k] = new_tp_splits[i]

            for i, states in enumerate(new_tp_states):
                for key, value in states.items():
                    states[key] = value.detach().clone()
                llm_save(os.path.join(saved_folder, f"model_tp{i}_pp{pp}.pt"), states)
                print(f"model_tp{i}_pp{pp}.pt saved in {saved_folder}.")

    def merge():
        for pp in range(pp_size):
            new_tp_states = [{} for _ in range(target_tp_size)]
            candidate_states = [defaultdict(list) for _ in range(target_tp_size)]
            for tp in range(tp_size):
                states_tp = llm_load(os.path.join(folder, f"model_tp{tp}_pp{pp}.pt"), map_location="cpu")
                for k, v in states_tp.items():
                    print("zlt tp key", k)
                    assert len(v.size()) < 3, "Only support 2D or 1D tensors."
                    candidate_states[tp // ratio][k].append(v)

            for i, states in enumerate(candidate_states):
                for k, v in states.items():
                    if "norm" in k:
                        new_tp_states[i][k] = v[0]
                        assert torch.equal(v[0], v[1]), f"{k} layers of pipeline {pp} are not equal."
                    elif k == "tok_embeddings.weight" or "wo" in k or "w2" in k or k == "embed_tokens.weight":
                        new_tp_states[i][k] = torch.concat(v, dim=column_dim)
                    elif k == "output.weight" or "w1" in k or "w3" in k or "wqkv" in k or "wq" in k or "wk" in k or "wv" in k:
                        new_tp_states[i][k] = torch.concat(v, dim=raw_dim)
                    else:
                        raise KeyError(f"Unknown key {k}.")

            for i, states in enumerate(new_tp_states):
                for key, value in states.items():
                    states[key] = value.detach().clone()
                llm_save(os.path.join(saved_folder, f"model_tp{i}_pp{pp}.pt"), states)
                print(f"model_tp{i}_pp{pp}.pt saved in {saved_folder}.")

    if tp_size < target_tp_size:
        split(split_maps)
    else:
        merge()


def adaptive_param_load(folder, exist_parts, current_part, exist_params, first=False, last=False):
    """Adaptively load the parameters of a model."""
    start, end = current_part[0]
    new_param_dict = {}
    key_index = 0

    for exist_id, exist_part in enumerate(exist_parts):
        exist_start, exist_end = exist_part[0]
        if start < end:
            if start < exist_end:
                exist_param = llm_load(os.path.join(folder, exist_params[exist_id]), map_location="cpu")
                exist_keys = list(exist_param.keys())

                if first and exist_start == 0:
                    if using_qwen:
                        new_param_dict["embed_tokens.weight"] = exist_param["embed_tokens.weight"]
                    else:
                        new_param_dict["tok_embeddings.weight"] = exist_param["tok_embeddings.weight"]
                if last and exist_end == exist_parts[-1][0][-1]:
                    new_param_dict["norm.weight"] = exist_param["norm.weight"]
                    new_param_dict["output.weight"] = exist_param["output.weight"]

                should_load_layer_index = range(start - exist_start, min(end, exist_end) - exist_start)
                for idx in should_load_layer_index:
                    for key in exist_keys:
                        print("zlt pp key", key)
                        if f"layers.{idx}." in key:
                            prefix = key.split(".")[:1]
                            prefix.append(str(key_index))
                            prefix.extend(key.split(".")[2:])
                            new_key = ".".join(prefix)
                            # print(f'{new_key} are loaded from {key} of {exist_params[exist_id]}')
                            new_param_dict[new_key] = exist_param[key]
                    key_index += 1
                start = exist_end
        else:
            break

    return new_param_dict


def convert_pp_size(folder, saved_folder, target_pp_size, model_size):
    """Convert the pipeline parallel size of a model."""
    pp_size, tp_size, model_fns = get_parallel_infos(folder)

    if pp_size > target_pp_size:
        assert pp_size % target_pp_size == 0, f"Cannot convert {pp_size} PP to {target_pp_size} PP."
    elif pp_size < target_pp_size:
        assert target_pp_size % pp_size == 0, f"Cannot convert {pp_size} PP to {target_pp_size} PP."
    else:
        if os.path.exists(saved_folder):
            shutil.rmtree(saved_folder)
        shutil.copytree(folder, saved_folder)
        print(f"Model has already been converted to {target_pp_size} PP. Copied to {saved_folder}.")
        return

    origin_pp_mapping = PIPELINE_SHARDING_INFO_QWEN[model_size][str(pp_size)]
    target_pp_mapping = PIPELINE_SHARDING_INFO_QWEN[model_size][str(target_pp_size)]

    origin_length = len(origin_pp_mapping)
    assert target_pp_size == len(
        target_pp_mapping
    ), f"Check the mapping of {model_size} {pp_size} PP in `PIPELINE_SHARDING_INFO`."

    for tp in range(tp_size):
        fn = model_fns[origin_length * tp : origin_length * (tp + 1)]
        for pp in range(target_pp_size):
            first = pp == 0
            last = pp == target_pp_size - 1
            new_param = adaptive_param_load(folder, origin_pp_mapping, target_pp_mapping[pp], fn, first, last)

            new_name = f"model_tp{tp}_pp{pp}.pt"
            for key, value in new_param.items():
                new_param[key] = value.detach().clone()
            llm_save(os.path.join(saved_folder, new_name), new_param)
            print(f"{new_name} saved in {saved_folder}.")


if __name__ == "__main__":

    args = parse_args()

    if "http_proxy" in os.environ:
        del os.environ["http_proxy"]
    if "https_proxy" in os.environ:
        del os.environ["https_proxy"]
    if "HTTP_PROXY" in os.environ:
        del os.environ["HTTP_PROXY"]
    if "HTTPS_PROXY" in os.environ:
        del os.environ["HTTPS_PROXY"]

    if not os.path.exists(args.target_model_path):
        os.makedirs(args.target_model_path)

    if args.target_tp_size > 0 and args.target_pp_size == 0:
        print("Converting TP size...")
        convert_tp_size(
            folder=args.origin_model_path, saved_folder=args.target_model_path, target_tp_size=args.target_tp_size
        )
    elif args.target_tp_size == 0 and args.target_pp_size > 0:
        print("Converting PP size...")
        convert_pp_size(
            folder=args.origin_model_path,
            saved_folder=args.target_model_path,
            target_pp_size=args.target_pp_size,
            model_size=args.model_size,
        )
    elif args.target_tp_size > 0 and args.target_pp_size > 0:
        print("Converting TP & PP size...\nFirst, converting TP size...")
        tmp_folder = "/dev/shm/convert_model_parallel_tmp"
        if os.path.exists(tmp_folder):
            shutil.rmtree(tmp_folder)
        os.makedirs(tmp_folder)
        convert_tp_size(folder=args.origin_model_path, saved_folder=tmp_folder, target_tp_size=args.target_tp_size)
        print("Second, converting PP size...")
        convert_pp_size(
            folder=tmp_folder,
            saved_folder=args.target_model_path,
            target_pp_size=args.target_pp_size,
            model_size=args.model_size,
        )
        shutil.rmtree(tmp_folder)  # free precious /dev/shm space
    else:
        raise ValueError("Please specify the target TP size or the target PP size.")

    print("Done!")
