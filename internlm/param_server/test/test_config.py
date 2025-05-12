import random

import torch

random.seed(0)


ps_servers = {
    0: "10.140.60.1",
    1: "10.140.60.2",
}
NUM_PS = len(ps_servers)
NUM_LAYERS = 82

MASTER_ADDR = "10.140.60.1"
MASTER_PORT = 55500
ZMQ_PORT = 55502
GRPC_PORT = 55501

UPDATE_METHOD = "async_buffer"  # or async_buffer

GRACE_TIME_IN_SECOND = 60

PS_OFFLINE_THRESHOLD = 80  # second
GROUPS_OFFLINE_THRESHOLD = 80  # second
# PS_OFFLINE_THRESHOLD = 5 * 60 # second
# GROUPS_OFFLINE_THRESHOLD = 5 * 60 # second
ALIVE_CHECK_INTERVAL = 60  # second
DUMP_STATE_INTERVAL = 60  # second
HEARTBEAT_INTERVAL = 60  # second


MASTER_VERSION_FILE = "./ps_version.txt"
CONSUME_TOKEN_WHEN_PS_SAVE_CKPT = 1024 * 1024 * 1024  # 1B

master_server = f"{MASTER_ADDR}:{MASTER_PORT}"
grpc_servers = {ps_id: f"{server}:{GRPC_PORT}" for ps_id, server in ps_servers.items()}
zmq_servers = {ps_id: f"tcp://{server}:{ZMQ_PORT}" for ps_id, server in ps_servers.items()}

model = dict(
    dtype=torch.bfloat16,
    num_layers=NUM_LAYERS,
    hidden_size=10240,
    vocab_size=65632,
    mlp_ratio=2.675,
)


def get_chunks(num_layers, num_chunks):
    # 初始化层索引列表
    layer_idxs = list(range(num_layers))
    first_layer = 0
    last_layer = num_layers - 1

    # 分离第一层和最后一层
    middle_layers = layer_idxs[1:-1]
    random.shuffle(middle_layers)  # 随机打乱中间层的顺序

    # 计算每个块的基本大小和余数
    chunk_size = num_layers // num_chunks
    # remainder = num_layers % num_chunks

    # 分配第一块，确保包含层 0，并且块大小为 chunk_size
    first_chunk = [first_layer]
    first_chunk += middle_layers[: chunk_size - 1]

    # 分配最后一块，确保包含层 num_layers-1，并且块大小为 chunk_size
    last_chunk = [last_layer]
    last_chunk += middle_layers[chunk_size - 1 : 2 * (chunk_size - 1)]

    # 剩余的层
    remaining_layers = middle_layers[2 * (chunk_size - 1) :]

    # 计算剩余的块数
    remaining_chunks = num_chunks - 2

    # 分配剩余的层到中间的块中
    other_chunks = []
    start = 0
    for i in range(remaining_chunks):
        # 计算当前块应该分配的层数
        current_size = (len(remaining_layers) // remaining_chunks) + (
            1 if i < (len(remaining_layers) % remaining_chunks) else 0
        )
        # 从剩余的层中分配当前块
        chunk = remaining_layers[start : start + current_size]
        other_chunks.append(chunk)
        start += current_size

    # 组合所有块
    chunks = [first_chunk] + other_chunks + [last_chunk]

    random.shuffle(chunks)

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
    mlp_ratio = config.get("mlp_ratio")

    mlp_hidden_features = int(hidden_size * mlp_ratio)

    # Define parameter shapes
    param_shapes = {
        "attention_norm.weight": (hidden_size,),  # Layer 23 attention norm weight
        "ffn_norm.weight": (hidden_size,),  # Layer 23 feedforward norm weight
        "attention.wq.weight": (hidden_size, hidden_size),  # Query weight for attention
        "attention.wk.weight": (hidden_size, hidden_size),  # Key weight for attention
        "attention.wv.weight": (hidden_size, hidden_size),  # Value weight for attention
        "attention.wo.weight": (hidden_size, hidden_size),  # Output projection weight for attention
        "feed_forward.w1.weight": (mlp_hidden_features, hidden_size),  # MLP first layer weight
        "feed_forward.w3.weight": (mlp_hidden_features, hidden_size),  # MLP third layer weight
        "feed_forward.w2.weight": (hidden_size, mlp_hidden_features),  # MLP second layer weight
    }

    return param_shapes


layer_chunks = get_chunks(NUM_LAYERS, NUM_PS)
param_shapes = get_param_shapes(model)
