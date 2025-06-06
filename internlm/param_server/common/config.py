import random

import torch

import os


random.seed(0)


ps_servers = {
    0: "10.10.41.41",
    1: "10.10.41.42",
}

USE_DLSLIME_RDMA_TRANSFER=True
NUM_PS = len(ps_servers)

MASTER_ADDR = "10.10.41.41"
MASTER_PORT = 55500
ZMQ_PORT = 55580
GRPC_PORT = 55577
GRPC_TIMEOUT = 10.0  # second

UPDATE_METHOD = "partial_sync"  # sync or partial_sync or async_buffer

GRACE_TIME_IN_SECOND = 60
PARTIAL_SYNC_RATE_THRESHOLD = 0.8
PARTIAL_SYNC_WAIT_TIME = 300  # second
SOCKET_TIME_IN_SECOND = 300

PS_OFFLINE_THRESHOLD = 80  # second
GROUPS_OFFLINE_THRESHOLD = 80  # second
# PS_OFFLINE_THRESHOLD = 5 * 60 # second
# GROUPS_OFFLINE_THRESHOLD = 5 * 60 # second
ALIVE_CHECK_INTERVAL = 25  # second
DUMP_STATE_INTERVAL = 60  # second
HEARTBEAT_INTERVAL = 25  # second


CONSUME_TOKEN_WHEN_PS_SAVE_CKPT = 1024 * 1024 * 1024 * 1024  # 1B

master_server = f"{MASTER_ADDR}:{MASTER_PORT}"
grpc_servers = {ps_id: f"{server}:{GRPC_PORT}" for ps_id, server in ps_servers.items()}
zmq_servers = {ps_id: f"tcp://{server}:{ZMQ_PORT}" for ps_id, server in ps_servers.items()}


model_type = os.environ.get("MODEL_TYPE", None)
MODEL_TYPE_LIST = ["INTERNLM_2_7B", "QWEN_2_7B", "LLAMA_2_7B", "QWEN_3_30B"]
MODEL_PARAM_DICT = dict()
MODEL_PARAM_DICT["INTERNLM_2_7B"] = dict()
MODEL_PARAM_DICT["INTERNLM_2_7B"]["NUM_LAYERS"] = 32
MODEL_PARAM_DICT["INTERNLM_2_7B"]["MLP_RATIO"] = 3.5
MODEL_PARAM_DICT["INTERNLM_2_7B"]["HIDDEN_SIZE"] = 4096
MODEL_PARAM_DICT["INTERNLM_2_7B"]["NUM_ATTENTION_HEAD"] = 32
MODEL_PARAM_DICT["INTERNLM_2_7B"]["NUM_KV_ATTENTION_HEAD"] = 8
MODEL_PARAM_DICT["INTERNLM_2_7B"]["VOCAB_SIZE"] = 92544
MODEL_PARAM_DICT["INTERNLM_2_7B"]["HEAD_DIM"] = 128


MODEL_PARAM_DICT["QWEN_3_30B"] = dict()
MODEL_PARAM_DICT["QWEN_3_30B"]["NUM_LAYERS"] = 48
MODEL_PARAM_DICT["QWEN_3_30B"]["MLP_RATIO"] = 768 / 2048
MODEL_PARAM_DICT["QWEN_3_30B"]["HIDDEN_SIZE"] = 2048
MODEL_PARAM_DICT["QWEN_3_30B"]["NUM_ATTENTION_HEAD"] = 32
MODEL_PARAM_DICT["QWEN_3_30B"]["NUM_KV_ATTENTION_HEAD"] = 4
MODEL_PARAM_DICT["QWEN_3_30B"]["VOCAB_SIZE"] = 151936
MODEL_PARAM_DICT["QWEN_3_30B"]["HEAD_DIM"] = 128


if model_type not in MODEL_TYPE_LIST:
    model_type = "INTERNLM_2_7B"
model_type = "QWEN_3_30B"
NUM_LAYERS = MODEL_PARAM_DICT[model_type]["NUM_LAYERS"]
MLP_RATIO = MODEL_PARAM_DICT[model_type]["MLP_RATIO"]
HIDDEN_SIZE = MODEL_PARAM_DICT[model_type]["HIDDEN_SIZE"]
NUM_ATTENTION_HEAD = MODEL_PARAM_DICT[model_type]["NUM_ATTENTION_HEAD"]
NUM_KV_ATTENTION_HEAD = MODEL_PARAM_DICT[model_type]["NUM_KV_ATTENTION_HEAD"]
VOCAB_SIZE = MODEL_PARAM_DICT[model_type]["VOCAB_SIZE"]
HEAD_DIM = MODEL_PARAM_DICT[model_type]["HEAD_DIM"]

model = dict(
    dtype=torch.bfloat16,
    num_layers=NUM_LAYERS,
    hidden_size=HIDDEN_SIZE,
    vocab_size=VOCAB_SIZE,
    num_attention_heads=NUM_ATTENTION_HEAD,
    num_kv_attention_heads=NUM_KV_ATTENTION_HEAD,
    mlp_ratio=MLP_RATIO,
    head_dim=HEAD_DIM,
)


def get_chunks(num_layers, num_chunks):
    layer_idxs = list(range(num_layers))
    chunk_size = num_layers // num_chunks
    remaining_size = num_layers % num_chunks

    chunks = []
    start_idx = 0
    for _ in range(num_chunks - remaining_size):
        chunks.append(layer_idxs[start_idx : start_idx + chunk_size])
        start_idx += chunk_size

    if remaining_size == 0:
        return chunks

    chunk_size += 1
    for _ in range(num_chunks - remaining_size, num_chunks):
        chunks.append(layer_idxs[start_idx : start_idx + chunk_size])
        start_idx += chunk_size

    if num_chunks > 2:
        chunks[1][-1], chunks[-1][-1] = chunks[-1][-1], chunks[1][-1]

    return chunks


def get_param_shapes(config):
    """
    Given a configuration dictionary for a LLaMA2 model, this function computes and outputs
    the shapes of common parameters such as w1, w2, w3, wq, wk, wv, etc.

    Args:
        config (dict): Dictionary containing the model configuration.

    Returns:
        dict: A dictionary mapping parameter names to their shapes.
    """
    # Extract config parameters
    hidden_size = config.get("hidden_size")
    num_attention_heads = config.get("num_attention_heads")
    num_kv_attention_heads = config.get("num_kv_attention_heads")
    mlp_ratio = config.get("mlp_ratio")
    vocab_size = config.get("vocab_size")

    mlp_hidden_features = int(hidden_size * mlp_ratio)
    # multiple_of = 256
    # mlp_hidden_features = (mlp_hidden_features + 255) // 256 * 256
    # Compute per-head size
    attn_head_dim = config.get("head_dim")
    q_dim = attn_head_dim * num_attention_heads
    kv_dim = attn_head_dim * num_kv_attention_heads

    # Define parameter shapes
    param_shapes = {
        "attention_norm.weight": (hidden_size,),  # Layer 23 attention norm weight
        "ffn_norm.weight": (hidden_size,),  # Layer 23 feedforward norm weight
        "attention.wqkv.weight": (int(q_dim + 2 * kv_dim), hidden_size),  # Combined QKV weight for attention
        "attention.wq.weight": (q_dim, hidden_size),  # Query weight for attention
        "attention.wk.weight": (kv_dim, hidden_size),  # Key weight for attention
        "attention.wv.weight": (kv_dim, hidden_size),  # Value weight for attention
        "attention.wo.weight": (hidden_size, q_dim),  # Output projection weight for attention
        "feed_forward.w1.weight": (int(mlp_hidden_features), hidden_size),  # MLP first layer weight
        "feed_forward.w3.weight": (int(mlp_hidden_features), hidden_size),  # MLP third layer weight
        "feed_forward.w2.weight": (hidden_size, int(mlp_hidden_features)),  # MLP second layer weight
        "feed_forward.fused_w1_w3.weight": (int(mlp_hidden_features) * 2, hidden_size),  # Fused first and third layer weight
        "norm.weight": (hidden_size,),  # Final normalization layer weight
        "tok_embeddings.weight": (vocab_size, hidden_size),  # Token embedding matrix
        "output.weight": (vocab_size, hidden_size),  # Output projection weight
        "embed_tokens.weight": (vocab_size, hidden_size), # Token embedding matrix of qwen2
        "attention.wq.bias": (q_dim,),
        "attention.wk.bias": (kv_dim,),
        "attention.wv.bias": (kv_dim,),
        "attention.wo.bias": (q_dim,),
        "attention.k_norm.weight": (attn_head_dim,),
        "attention.q_norm.weight": (attn_head_dim,),
        "feed_forward.moe_layer.gate.wg.weight": (attn_head_dim, hidden_size),
        "w1.weight": (mlp_ratio * hidden_size, hidden_size),
        "w2.weight": (hidden_size, mlp_ratio * hidden_size),
        "w3.weight": (mlp_ratio * hidden_size, hidden_size),
    }

    return param_shapes


ckpt = dict(
    auto_resume=False,
    # load_ckpt_path="/data/InternEvo-psserver/20B_ckpt/internlm2/1_merged/model_tp0_pp0.pt",
    # load_ckpt_path="/datapool/caikun/ckpt/internlm2_7b_ckpt/model_tp0_pp0.pt",
    load_ckpt_path="/datapool/caikun/ckpt/qwen3_30b_a3b/model_wp0_pp0.pt",
    #load_ckpt_path="/datapool/zhanglantian/InternEvo/tools/zlt_test_convert/test.pt",
    save_ckpt_path="./ps_ckpt",
)
layer_chunks = get_chunks(NUM_LAYERS, NUM_PS)
param_shapes = get_param_shapes(model)
optimizer = dict(
    name="nesterov",
    lr=0.9,
    momentum=0.8,
)

grad = dict(
    use_ewma_outlier=False,
    ewma=dict(
        alpha=0.02,
        warmup_steps=5,
        base_threshold=3,
    ),
    use_weighted_avg=True,
    clip_pseudo_grad=1.0,
)
