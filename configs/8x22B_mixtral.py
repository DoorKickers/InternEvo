JOB_NAME = "22b_moe_mixtral"
model_type = "MIXTRALMOE"
DO_ALERT = False

SEQ_LEN = 4096
HIDDEN_SIZE = 6144
NUM_ATTENTION_HEAD = 48
NUM_KV_ATTENTION_HEAD = 8
MLP_RATIO = 8 / 3
NUM_LAYER = 56
VOCAB_SIZE = 32000

MODEL_ONLY_FOLDER = "local:llm_ckpts/xxxx"
# Ckpt folder format:
# fs: 'local:/mnt/nfs/XXX'
SAVE_CKPT_FOLDER = "local:llm_ckpts"
LOAD_CKPT_FOLDER = "local:llm_ckpts/49"

# boto3 Ckpt folder format:
# import os
# BOTO3_IP = os.environ["BOTO3_IP"] # boto3 bucket endpoint
# SAVE_CKPT_FOLDER = f"boto3:s3://model_weights.{BOTO3_IP}/internlm"
# LOAD_CKPT_FOLDER = f"boto3:s3://model_weights.{BOTO3_IP}/internlm/snapshot/1/"
CHECKPOINT_EVERY = 50
ckpt = dict(
    enable_save_ckpt=False,  # enable ckpt save.
    save_ckpt_folder=SAVE_CKPT_FOLDER,  # Path to save training ckpt.
    # load_ckpt_folder= dict(path=MODEL_ONLY_FOLDER, content=["model"], ckpt_type="normal"),
    load_ckpt_folder="local:llm_ckpts/",
    # 'load_ckpt_info' setting guide:
    # 1. the 'path' indicate ckpt path,
    # 2. the 'content‘ means what states will be loaded, support: "model", "sampler", "optimizer", "scheduler", "all"
    # 3. the ’ckpt_type‘ means the type of checkpoint to be loaded, support: "internevo", "hf", or other custom-defined
    # load function such as "llama"
    load_ckpt_info=dict(path=MODEL_ONLY_FOLDER, content=("model",), ckpt_type="internevo"),
    # 'auto_resume' is designed to automatically load the latest checkpoint from 'save_ckpt_folder' when encountering
    # training interruptions/hangs caused by hardware failures, using a scheduling system (such as k8s/slurm)
    # with an automatic restart mechanism upon training reboot.
    # Please be aware that if `auto_resume` is not set (its default value is True), it will not load the checkpoint
    # path specified in `load_ckpt_info` by default.
    # If you want to initialize your model weights from another model, you must set `auto_resume` to False.
    # If you want to train from scratch, please set `auto_resume` to False and 'load_ckpt_info' to None.
    auto_resume=True,
    checkpoint_every=CHECKPOINT_EVERY,
    async_upload=True,  # async ckpt upload. (only work for boto3 ckpt)
    async_upload_tmp_folder="/dev/shm/internlm_tmp_ckpt/",  # path for temporarily files during asynchronous upload.
    oss_snapshot_freq=int(CHECKPOINT_EVERY / 2),  # snapshot ckpt save frequency.
)

TRAIN_FOLDER = None  # "/path/to/dataset"
VALID_FOLDER = None  # "/path/to/dataset"
data = dict(
    seq_len=SEQ_LEN,
    # micro_num means the number of micro_batch contained in one gradient update
    micro_num=4,
    # packed_length = micro_bsz * SEQ_LEN
    micro_bsz=2,
    # defaults to the value of micro_num
    valid_micro_num=4,
    # defaults to 0, means disable evaluate
    valid_every=50,
    pack_sample_into_one=False,
    total_steps=50000,
    skip_batches="",
    # rampup_batch_size (str): A string with three space-separated integers representing the
    #       starting batch size, the increment, and the number of steps between
    #       each increment. For example, "192 24 8" means that the batch size (micro_num)
    #       starts at 192 and increases by 24 every 8 steps. Defaults to None.
    #       (IMPORTANT): The interval step size is 'micro_bsz'.
    rampup_batch_size="",
    # Datasets with less than 50 rows will be discarded
    min_length=50,
    train_folder=TRAIN_FOLDER,
    valid_folder=VALID_FOLDER,
    empty_cache_and_diag_interval=200,
    diag_outlier_ratio=1.1,
)

grad_scaler = dict(
    fp16=dict(
        # the initial loss scale, defaults to 2**16
        initial_scale=2**16,
        # the minimum loss scale, defaults to None
        min_scale=1,
        # the number of steps to increase loss scale when no overflow occurs
        growth_interval=1000,
    ),
    # the multiplication factor for increasing loss scale, defaults to 2
    growth_factor=2,
    # the multiplication factor for decreasing loss scale, defaults to 0.5
    backoff_factor=0.5,
    # the maximum loss scale, defaults to None
    max_scale=2**24,
    # the number of overflows before decreasing loss scale, defaults to 2
    hysteresis=2,
)

hybrid_zero_optimizer = dict(
    # Enable low_level_optimzer overlap_communication
    overlap_sync_grad=False,
    overlap_sync_param=False,
    # bucket size for nccl communication params
    reduce_bucket_size=512 * 1024 * 1024,
    # grad clipping
    clip_grad_norm=1.0,
)

loss = dict(
    label_smoothing=0,
    moe_loss_coeff=0.001,
)

adam = dict(
    lr=1e-4,
    adam_beta1=0.9,
    adam_beta2=0.95,
    adam_beta2_c=0,
    adam_eps=1e-8,
    weight_decay=0.01,
)

lr_scheduler = dict(
    total_steps=data["total_steps"],
    init_steps=0,  # optimizer_warmup_step
    warmup_ratio=0.01,
    eta_min=1e-5,
    last_epoch=-1,
)

beta2_scheduler = dict(
    init_beta2=adam["adam_beta2"],
    c=adam["adam_beta2_c"],
    cur_iter=-1,
)

use_fp32_norm = False
model = dict(
    checkpoint=False,  # The proportion of layers for activation aheckpointing, the optional value are True/False/[0-1]
    num_attention_heads=NUM_ATTENTION_HEAD,
    num_kv_attention_heads=NUM_KV_ATTENTION_HEAD,
    max_position_embeddings=65536,
    embed_split_hidden=True,
    vocab_size=VOCAB_SIZE,
    embed_grad_scale=1,
    parallel_output=True,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYER,
    qkv_bias=False,
    o_bias=False,
    mlp_ratio=MLP_RATIO,
    apply_post_layer_norm=False,
    dtype="torch.bfloat16",  # Support: "torch.float16", "torch.half", "torch.bfloat16", "torch.float32", "torch.tf32"
    norm_type="rmsnorm",
    layer_norm_epsilon=1e-5,
    use_flash_attn=True,
    # Whether the odd and even columns of the query and key in the model are normally interleaved.
    # If it's True, the model's odd and even columns are normally ordered; if it's False,
    # it means that the model has prematurely concatenated all odd columns and even columns in front
    # and back, in order to improve the RoPE's computational efficiency.
    # Example:
    # qk_interleaved = True: q[-1] = [q1,q2,q3,q4,q5,q6,...], k[-1] = [k1,k2,k3,k4,k5,k6,...]
    # qk_interleaved = False: q[-1] = [q1,q3,q5,...,q2,q4,q6,...], k[-1] = [k1,k3,k5,...,k2,k4,k6,...]
    qk_interleaved=False,
    use_sliding_window=False,
    rope_base=1000000,
    num_chunks=1,  # if num_chunks > 1, interleaved pipeline scheduler is used.
    moe_type="GShard",  # Support: "GShard", "MegaBlock", "MegaBlock-D", "Dropless"
    num_experts=8,
    top_k=2,
)
"""
zero1 parallel (dict):
    1. size: int
        * if size <= 0, the size of the zero process group is equal to the size of the dp process group,
            so parameters will be divided within the range of dp.
        * if size == 1, zero is not used, and all dp groups retain the full amount of model parameters.
        * if size > 1 and size <= dp world size, the world size of zero is a subset of dp world size.
        For smaller models, it is usually a better choice to split the parameters within nodes with a setting <= 8.
tensor parallel (dict):
    1. size: int, the size of tensor parallel.
    2. mode: str, the tensor parallel mode, should be in ['mtp', 'msp', 'fsp', 'isp'],
        defaults to 'mtp', means the pure megatron tensor parallel without sequence parallel.
        msp: megatron tensor parallel with sequence parallel, sequence parallel size = tensor parallel size.
        fsp: tensor parallel by flash-attn with sequence parallel, sequence parallel size = tensor parallel size.
        isp: customed intern sequence parallel without tensor parallel, can be used with weight parallel.
pipeline parallel (dict):
    1. size: int, the size of pipeline parallel.
    2. interleaved_overlap: bool, enable/disable communication overlap when using interleaved pipeline scheduler,
        defaults to False.
weight parallel (dict):
    1. size: int, the size of weight parallel.
    2. overlap: bool, enable/disable all_gather/reduce_scatter communication overlap, defaults to False.
expert parallel (dict):
    1. size: int
        * if size <= 0, ep size equals to dp size, but if the number of experts is smaller than dp size, set ep size
            to be the number of experts to make sure each device has one expert.
        * if size == 1, all experts are placed in each device, running as dp-only.
        * if size > 1, all experts are placed in k devices and each device has n/k experts, where n is the total
            number of experts and k = size.
expert weight parallel (dict):
    1. size: int, the size of weight parallel for expert module, distinct with global weight parallel size.
    2. overlap: bool, enable/disable all_gather/reduce_scatter communication overlap, defaults to False.
"""
parallel = dict(
    zero1=dict(size=-1),
    tensor=dict(size=1, mode="mtp"),
    pipeline=dict(size=1, interleaved_overlap=True),
    weight=dict(size=1, overlap=True),
    expert=dict(size=-1, no_tp=False),
    expert_weight=dict(size=1, overlap=True),
)

cudnn_deterministic = False
cudnn_benchmark = False

monitor = dict(
    # feishu alert configs
    alert=dict(
        enable_feishu_alert=DO_ALERT,
        feishu_alert_address=None,  # feishu webhook to send alert message
        light_monitor_address=None,  # light_monitor address to send heartbeat
        alert_file_path=f"llm_alter/{JOB_NAME}_alert.log",
    ),
    tensorboard=dict(
        queue_max_length=10,
    ),
)
