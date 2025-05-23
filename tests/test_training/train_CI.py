#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import os

# pylint: disable=C0413,W0612,W0611
import socket
import sys
import time
import traceback
from functools import partial

import torch
import torch.distributed as dist

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../"))
sys.path.append(project_root)

import internlm  # noqa: E402
from internlm.checkpoint import CheckpointManager  # noqa: E402
from internlm.core.context import ParallelMode  # noqa: E402
from internlm.core.context import global_context as gpc  # noqa: E402
from internlm.core.trainer import Trainer, TrainState  # noqa: E402
from internlm.data import (  # noqa: E402
    build_train_loader_with_data_type,
    build_valid_loader_with_data_type,
)
from internlm.eval.evaluation import evaluate_on_val_dls  # noqa: E402
from internlm.initialize import initialize_distributed_env  # noqa: E402
from internlm.model.losses import InternLoss  # noqa: E402
from internlm.model.metrics import AccPerplex, SchedulerMetricHook  # noqa: E402
from internlm.monitor import (  # noqa: E402
    initialize_monitor_manager,
    send_alert_message,
)
from internlm.monitor.monitor import monitor_manager as mm  # noqa: E402
from internlm.train import (  # noqa: E402
    initialize_llm_profile,
    initialize_model_and_parallel_communicator,
    initialize_optimizer,
    record_current_batch_training_metrics,
)
from internlm.utils.common import (  # noqa: E402
    BatchSkipper,
    get_current_device,
    get_megatron_flops,
    launch_time,
    parse_args,
)
from internlm.utils.gputest import empty_cache_and_diag  # noqa: E402
from internlm.utils.logger import get_logger  # noqa: E402
from internlm.utils.megatron_timers import megatron_timer as timer  # noqa: E402
from internlm.utils.parallel import get_parallel_log_file_name  # noqa: E402
from internlm.utils.simple_memory_profiler import SimpleMemoryProfiler  # noqa: E402
from internlm.utils.writer import Writer  # noqa: E402

# global llm logger
logger = get_logger(__file__)


def fuse_wqkv(key, state_dict) -> None:  # pylint: disable=W0613
    prefix = key.rstrip(".Wqkv.weight")
    wq_name, wk_name, wv_name = (
        f"{prefix}.wq.weight",
        f"{prefix}.wk.weight",
        f"{prefix}.wv.weight",
    )

    wq, wk, wv = state_dict.pop(wq_name), state_dict.pop(wk_name), state_dict.pop(wv_name)
    state_dict[key] = torch.cat([wq, wk, wv], dim=0)


def check_model_weights(model, ckpt_path, total_equal=False):
    model = model.model
    model1_dict = torch.load(ckpt_path, map_location="cuda")
    model2_dict = model.state_dict()

    copy_of_ordered_dict = model2_dict.copy()

    for key in copy_of_ordered_dict.keys():
        if "wqkv" in key:
            model2_dict[key.replace("wqkv", "Wqkv")] = model2_dict.pop(key)
            key = key.replace("wqkv", "Wqkv")

        if key not in model1_dict:
            if "Wqkv" in key:
                fuse_wqkv(key, model1_dict)
            else:
                assert False, f"Error: The key {key} for current model dose not exist in standard ckpt!"

    for key in model1_dict.keys():
        if key in model2_dict:
            tensor1 = model1_dict[key]
            tensor2 = model2_dict[key]
            if total_equal:
                assert torch.equal(tensor1, tensor2), "model weights are not equal"
            else:
                assert torch.allclose(tensor1, tensor2, rtol=3e-2, atol=3e-2), "model weights are not close"
        else:
            if gpc.is_rank_for_log():
                logger.warning(f"The old key {key} no longer exists!")

    if gpc.is_rank_for_log():
        logger.info("Weight check passed")


def main(args):
    very_begining_time = time.time()
    # init setting
    skip_batches = gpc.config.data.skip_batches
    total_steps = gpc.config.data.total_steps
    valid_every = gpc.config.data.valid_every
    label_smoothing = gpc.config.loss.label_smoothing

    get_tflops_func = partial(
        get_megatron_flops,
        checkpoint=gpc.config.model.checkpoint,
        seq_len=gpc.config.data["seq_len"],
        hidden_size=gpc.config.model.hidden_size,
        num_layers=gpc.config.model.num_layers,
        vocab_size=gpc.config.model.vocab_size,
        global_batch_size=gpc.config.data.micro_bsz * gpc.config.data.micro_num * gpc.get_world_size(ParallelMode.DATA),
        global_world_size=gpc.get_world_size(ParallelMode.GLOBAL),
        mlp_ratio=gpc.config.model["mlp_ratio"],
    )

    # get and broadcast current time
    current_time = launch_time()
    objs = [current_time]
    dist.broadcast_object_list(objs, src=0)
    current_time = objs[0]

    # initialize model
    model , _ = initialize_model_and_parallel_communicator()

    with open(args.config, "r") as f:
        config_lines = f.readlines()

    # initialize loss function
    criterion = InternLoss(parallel_output=True, label_smoothing=label_smoothing)

    # initialize the train and validation data loader
    train_dl, dataset_types = build_train_loader_with_data_type()
    val_dls = build_valid_loader_with_data_type()

    # initialize and resume train state
    train_state = TrainState(gpc.config, train_dl.batch_sampler)

    optimizer, beta2_scheduler, lr_scheduler = initialize_optimizer(model=model)

    ckpt_manager = CheckpointManager(
        ckpt_config=gpc.config.ckpt,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_dl=train_dl,
        model_config=gpc.config.model,
        model_config_file="".join(config_lines),
        feishu_address=gpc.config.monitor.alert.feishu_alert_address,
    )

    # Loading other persistent training states.
    ckpt_manager.try_resume_training(train_state, current_time)

    # initialize customed llm writer
    writer = Writer(
        job_name=gpc.config.JOB_NAME,
        launch_time=current_time,
        file_name=get_parallel_log_file_name(),
        tensorboard_folder=gpc.config.tensorboard_folder,
        resume_tb_folder=train_state.resume_tb_folder,  # resume from ckpt.
        step_count=train_state.step_count,  # resume from ckpt.
        config=config_lines,
        logger=logger,
        enable_tb=gpc.config.enable_tb,
    )

    # initialize metric for calculating accuracy and perplexity
    metric = AccPerplex(
        device=get_current_device(),
        tp_pg=gpc.get_group(ParallelMode.TENSOR),
        dp_pg=gpc.get_group(ParallelMode.DATA),
        dataset_types=dataset_types,
    )

    # initialize trainer
    scheduler_hooks = [
        SchedulerMetricHook(
            metric=metric,
            skip=(
                gpc.is_using_parallel_mode(ParallelMode.PIPELINE)
                and hasattr(gpc.config.model, "num_chunks")
                and gpc.config.model.num_chunks > 1
                and gpc.config.parallel["pipeline"].get("interleaved_overlap", False)
            ),
        ),
    ]

    engine, scheduler = internlm.initialize_trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        lr_scheduler=lr_scheduler,
        beta2_scheduler=beta2_scheduler,
        scheduler_hooks=scheduler_hooks,
    )
    trainer = Trainer(engine, scheduler)

    # initialize simple memory profiler
    if args.profiling:
        memory_profiler = SimpleMemoryProfiler(
            model,
            optimizer.optim,
            log_folder=f"memory_trace/rank{gpc.get_global_rank()}_"
            + f"dp{gpc.get_local_rank(ParallelMode.DATA)}_"
            + f"tp{gpc.get_local_rank(ParallelMode.TENSOR)}",
        )
    else:
        memory_profiler = None

    # initialize the batch skipper
    batch_skipper = BatchSkipper(skip_batches)

    trainer.train()

    # transfer the train data loader into train data iterator
    # train_iter = iter(train_dl)

    # check model init weights
    if hasattr(gpc.config, "CHECK_INIT") and gpc.config.CHECK_INIT == 1:
        ckpt_name = (
            f"model"
            f"_tp{gpc.get_local_rank(ParallelMode.TENSOR)}"
            f"_pp{gpc.get_local_rank(ParallelMode.PIPELINE)}.pt"
        )
        ckpt_path = os.path.join(
            os.environ["share_path"], "quailty_assurance/7B_internlm2_init_dp=2_tp=2_pp=2_ckpt/init", ckpt_name
        )
        check_model_weights(model, ckpt_path, total_equal=True)
    with initialize_llm_profile(profiling=args.profiling, start_time=current_time) as prof:
        # start iterating the train data and begin training
        for batch_count in range(train_state.batch_count, total_steps):
            empty_cache_and_diag(batch_count, interval=gpc.config.data.empty_cache_and_diag_interval)
            start_time = time.time()
            timer("one-batch").start()

            # load batch data
            # batch, train_iter = load_new_batch(train_dl=train_dl, train_iter=train_iter, train_state=train_state)
            # pylint: disable=C0301
            batch_index = batch_count % 1000
            if batch_index == 0:
                data_local_rank = gpc.get_local_rank(ParallelMode.DATA)
                batch_step = (batch_count // 1000 + 1) * 1000
                data_path = os.path.join(
                    os.environ["share_path"],
                    "quailty_assurance/debug_Qiansanqiang_7B_v16",
                    f"dp-11{data_local_rank}/batch-{batch_step}.pt",
                )
                data_1000 = torch.load(data_path, map_location=torch.device("cpu"))
            batch = data_1000[batch_index]

            # record the consumed samples in training
            train_state.batch_count = batch_count
            train_state.num_consumed_samples_in_epoch += len(batch[1])
            if batch_skipper(batch_count):  # skip this batch
                if gpc.is_rank_for_log():
                    logger.info(f"Skip batch count:`{batch_count}`...")
                timer("one-batch").stop()
                continue

            # zero the grads of parameters
            trainer.zero_grad()
            # process data
            if batch[0].get("type_ids", None) is not None:
                metric.set_current_type_ids(type_ids=batch[0].pop("type_ids", None))

            # do forward and backward
            timer("fwd-bwd").start()

            moe_loss = None
            if hasattr(gpc.config.model, "num_experts"):
                _, _, loss, moe_loss = trainer.execute_schedule(
                    batch,
                    forward_only=False,
                    return_loss=True,
                    return_output_label=False,
                )
            else:
                _, _, loss = trainer.execute_schedule(
                    batch,
                    forward_only=False,
                    return_loss=True,
                    return_output_label=False,
                )
            timer("fwd-bwd").stop()

            # update parameters, and returns (success_update, grad_norm)
            trainer_result = trainer.step()
            assert trainer_result is not None

            success_update, grad_norm_groups = trainer_result
            if success_update:  # update parameters successfully
                train_state.step_count += 1
            else:
                train_state.inf_nan_skip_batches += 1  # record the amount of updating parameters unsuccessfully.
                if -1 in grad_norm_groups.values() and gpc.is_rank_for_log():  # -1 encodes a specific failure case
                    logger.warning(f"Warning: skip parameter update at step {batch_count}.")
                    send_alert_message(
                        address=gpc.config.monitor.alert.feishu_alert_address,
                        message=f"Warning: skip parameter update at step {batch_count}.",
                    )

            # calculate and record the training metrics, eg. loss, accuracy and so on.
            record_current_batch_training_metrics(
                get_tflops_func=get_tflops_func,
                logger=logger,
                writer=writer,
                success_update=success_update,
                batch_count=batch_count,
                batch=batch,
                train_state=train_state,
                optimizer=optimizer,
                beta2_scheduler=beta2_scheduler,
                engine=trainer.engine,
                very_begining_time=very_begining_time,
                start_time=start_time,
                loss=loss,
                moe_loss=moe_loss,
                grad_norm=grad_norm_groups,
                metric=metric,
            )

            timer("one-batch").stop()

            # evaluate on validation data loaders
            if valid_every > 0 and train_state.step_count % valid_every == 0:
                evaluate_on_val_dls(
                    trainer=trainer,
                    val_dls=val_dls,
                    writer=writer,
                    logger=logger,
                    step_count=train_state.step_count,
                )

            # check model weights
            if batch_count > 0 and batch_count % 100 == 0:
                ckpt_name = (
                    f"model"
                    f"_tp{gpc.get_local_rank(ParallelMode.TENSOR)}"
                    f"_pp{gpc.get_local_rank(ParallelMode.PIPELINE)}.pt"
                )
                ckpt_path = os.path.join(
                    os.environ["share_path"],
                    "quailty_assurance/7B_internlm2_init_dp=2_tp=2_pp=2_ckpt",
                    str(batch_count),
                    ckpt_name,
                )
                check_model_weights(model, ckpt_path)

            # checkpoint the training states in specific steps, which is determined by the args "checkpoint_every"
            # # save batch sampler that tracks the true consumed samples
            now_break = ckpt_manager.try_save_checkpoint(train_state)
            if now_break:
                break

            if memory_profiler is not None:
                memory_profiler.step()

            if batch_count % 2 == 0:
                prof.step()

    ckpt_manager.wait_async_upload_finish()


if __name__ == "__main__":
    args = parse_args()
    hostname = socket.gethostname()

    # initialize distributed environment
    initialize_distributed_env(config=args.config, launcher=args.launcher, master_port=args.port, seed=args.seed)
    assert hasattr(gpc, "config") and gpc.config is not None

    # initialize monitor manager context
    with initialize_monitor_manager(
        job_name=gpc.config.JOB_NAME, alert_address=gpc.config.monitor.alert.feishu_alert_address
    ):
        try:
            main(args)
        except AssertionError as e:
            logger.error(e)
            sys.exit(1)
        except Exception:
            logger.error(
                f"Raise exception from {hostname} with rank id: {gpc.get_global_rank()}\n{traceback.format_exc()}",
            )
            mm.monitor_exception(
                alert_address=gpc.config.monitor.alert.feishu_alert_address, excp_info=traceback.format_exc()
            )
            sys.exit(1)
