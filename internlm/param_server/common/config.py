import random

import torch

random.seed(0)


ps_servers = {
    0: "127.0.0.1",
}
NUM_PS = len(ps_servers)
NUM_LAYERS = 28

MASTER_ADDR = "127.0.0.1"
MASTER_PORT = 55500
ZMQ_PORT = 55502
GRPC_PORT = 55501
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

MLP_RATIO_7B = 2.6875
MLP_RATIO_20B = 8 / 3
MLP_RATIO_QWEN = 5.25
model = dict(
    dtype=torch.bfloat16,
    num_layers=NUM_LAYERS,
    hidden_size=3584,
    vocab_size=152064,
    num_attention_heads=28,
    num_kv_attention_heads=4,
    mlp_ratio=MLP_RATIO_QWEN,
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
    multiple_of = 256
    mlp_hidden_features = (mlp_hidden_features + 255) // 256 * 256
    # Compute per-head size
    attn_head_dim = hidden_size // num_attention_heads
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
        "attention.wo.weight": (hidden_size, hidden_size),  # Output projection weight for attention
        "feed_forward.w1.weight": (mlp_hidden_features, hidden_size),  # MLP first layer weight
        "feed_forward.w3.weight": (mlp_hidden_features, hidden_size),  # MLP third layer weight
        "feed_forward.w2.weight": (hidden_size, mlp_hidden_features),  # MLP second layer weight
        "feed_forward.fused_w1_w3.weight": (mlp_hidden_features * 2, hidden_size),  # Fused first and third layer weight
        "norm.weight": (hidden_size,),  # Final normalization layer weight
        "tok_embeddings.weight": (vocab_size, hidden_size),  # Token embedding matrix
        "output.weight": (vocab_size, hidden_size),  # Output projection weight
        "embed_tokens.weight": (vocab_size, hidden_size), # Token embedding matrix of qwen2
        "attention.wq.bias": (q_dim,),
        "attention.wk.bias": (kv_dim,),
        "attention.wv.bias": (kv_dim,),
        "attention.wo.bias": (hidden_size,),

    }

    return param_shapes


ckpt = dict(
    auto_resume=False,
    # load_ckpt_path="/data/InternEvo-psserver/20B_ckpt/internlm2/1_merged/model_tp0_pp0.pt",
    load_ckpt_path="/nvme/zhanglantian/new/my/InternEvo/ckpt_qwen2_7B_merged/model_tp0_pp0.pt",
    save_ckpt_path="./ps_ckpt",
)
layer_chunks = get_chunks(NUM_LAYERS, NUM_PS)
param_shapes = get_param_shapes(model)
optimizer = dict(
    name="nesterov",
    lr=0.7,
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