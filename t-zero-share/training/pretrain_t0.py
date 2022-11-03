# coding=utf-8

import argparse
import logging
import os
import random
from dataclasses import dataclass
from itertools import chain
from typing import Optional, Union
import csv
import math

import datasets
import torch
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    default_data_collator,
    DataCollatorForSeq2Seq,
    AdamW,
    SchedulerType,
    get_scheduler,
    set_seed,
)
from transformers.file_utils import PaddingStrategy
from templates import DatasetTemplates
from pretraining_utils import load_training_task_list, load_all_dataset


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="预训练t0")
    parser.add_argument("--train_task_list", type=str, default=None, required=True,
                        help="用于预训练的任务列表")
    parser.add_argument("--data_dir", type=str, default=None, required=True,
                        help="用于预训练的数据集列表（需要已经预处理过，每个prompt一个文件夹）")
    parser.add_argument("--train_file", type=str, default=None,
                        help="预训练文件，全部都处理好的")
    parser.add_argument("--eval_file", type=str, default=None,
                        help="预训练文件，全部都处理好的")
    parser.add_argument("--output_dir", type=str, default=None, required=True,
                        help="Where to store the results CSV and (TODO) optionally the final model.")

    parser.add_argument("--template_dir", type=str, default=None, required=True,
                        help="模板文件的路径")
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help=("Path to pretrained model or model identifier from huggingface.co/models. "
                              "The list of T0 variants can be found on `https://huggingface.co/bigscience/T0_3B`"), )

    parser.add_argument("--parallelize", action="store_true",
                        help="If passed, will call `model.parallelize` which splits the model on all GPUs available (model parallelism). "
                             "Note that this feature is still experimental in HF Transformers.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8,
                        help="Batch size (per device) for the evaluation dataloader. Will be multiplied by the number of answer choices.")

    parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                        help="Batch size (per device) for the training dataloader.")

    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4,
                        help="Initial learning rate (after the potential warmup period) to use.")

    parser.add_argument("--num_train_epochs", type=int, default=10,
                        help="Total number of training epochs to perform.")

    parser.add_argument("--max_train_steps", type=int, default=None,
                        help="Total number of training steps to perform. If provided, overrides num_train_epochs.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--eval_interval_steps", type=int, default=10000,
                        help="每隔多少步eval一次")
    parser.add_argument("--start_eval_steps", type=int, default=100000,
                        help="从什么时候开始eval")
    # T0训练的时候包含eos，但是t0没有包含
    parser.add_argument("--input_eos", action="store_true",
                        help="T0 was trained without EOS in its input sequences, which is the default in this script."
                             "However, T5 was pretrained with EOS in its input sequences. See README for more info.")
    parser.add_argument("--debug", action="store_true",
                        help="Activate debug mode and run training only with a subset of data.")
    # Weights & Biases, 一个替代tensorboard的库，有空可以看看
    parser.add_argument("--wandb_proj", type=str, default=None,
                        help="Project name for Weights & Biases. By default, W&B is disabled.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Especially important for few-shot example sampling.")
    parser.add_argument("--max_length", type=int, default=1024,
                        help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
                             " sequences shorter will be padded if `--pad_to_max_lengh` is passed.")
    parser.add_argument("--target_max_length", type=int, default=256,
                        help="Target max length. Sequences longer than this will be truncated.")
    parser.add_argument("--pad_to_max_length", action="store_true",
                        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.")

    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay for the AdamW optimizer.")
    parser.add_argument("--lr_scheduler_type", type=SchedulerType, default="linear",
                        help="The scheduler type to use.",
                        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant",
                                 "constant_with_warmup"])
    parser.add_argument("--num_warmup_steps", type=int,
                        default=0, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    set_seed(args.seed)

    # Initialize the accelerator. We will let the accelerator handle device placement for us.
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO, )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    # 在本机主进程设置datasets的日志，i dont care
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # Handle the output directory creation
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    train_task_list = load_training_task_list(args.train_task_list)

    with accelerator.main_process_first():
        raw_train_dataset, raw_eval_dataset = load_all_dataset(args, logger, train_task_list,
                                                               args.data_dir, args.template_dir)
        logger.info(f'raw_train_dataset: {raw_train_dataset}')
        logger.info(f'raw_eval_dataset: {raw_eval_dataset}')

    accelerator.wait_for_everyone()

    if args.debug:
        raw_train_dataset = raw_train_dataset.select(range(min(100, len(raw_train_dataset))))
        raw_eval_dataset = raw_eval_dataset.select(range(min(100, len(raw_eval_dataset))))

    # column_names = raw_eval_dataset.column_names

    # 读模型
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path, config=config)

    # padding = "max_length" if args.pad_to_max_length else False

    for index in range(0, 3):
        logger.debug(f"Sample {index} of the training set: {raw_train_dataset[index]}.")
    for index in range(0, 3):
        logger.debug(f"Sample {index} of the evaluation set: {raw_eval_dataset[index]}.")

    train_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if accelerator.use_fp16 else None
    )
    train_dataloader = DataLoader(
        raw_train_dataset,
        shuffle=True,
        collate_fn=train_collator,
        batch_size=args.per_device_train_batch_size
    )

    # eval的collator
    eval_collator = default_data_collator
    eval_dataloader = DataLoader(raw_eval_dataset, collate_fn=eval_collator, batch_size=args.per_device_eval_batch_size)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    num_gpus = torch.cuda.device_count()  # 这里对distributed进行了修正
    logger.info(f'num_gpus: {num_gpus}')
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps / num_gpus)
    # 这个训练步数没有考虑到多卡的情况？
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # 对model进行parallel
    if args.parallelize:
        assert num_gpus > 1, "You need at least 2 GPUs to use `model.parallelize()`."
        model.parallelize()
        optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
            optimizer, train_dataloader, eval_dataloader)
    else:
        # dataloader的长度在prepare 后变变 total_batch_num // n_gpu
        model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader)

    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(raw_train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    global_steps = 0

    train_loss_list = []
    min_eval_loss = 999999
    accelerator.wait_for_everyone()
    for epoch in range(1, args.num_train_epochs + 1):
        model.train()
        # 一次跑一个epoch
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / args.gradient_accumulation_steps
            train_loss_list.append(loss.item())
            accelerator.backward(loss)
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                global_steps += 1
                # loss = loss.item()

                # 注意这里一个step有8个gpu batch被优化了，所以max_train_steps算的是错的(没有考虑到distributed的情况)
                logger.info(f'epoch = {epoch}, step = {global_steps}, loss = {loss.item()}')

            if global_steps % 50 == 0 and accelerator.is_main_process:
                tqdm.write(f"epoch = {epoch}, step = {global_steps}, loss = {sum(train_loss_list)}")
                train_loss_list = []

            if global_steps >= args.max_train_steps:
                break

            if global_steps > args.start_eval_steps and global_steps % args.eval_interval_steps == 0:
                # start eval
                model.eval()
                eval_loss = eval_during_training(args, accelerator, eval_dataloader, model)

                if eval_loss < min_eval_loss:
                    min_eval_loss = eval_loss

                    # save model
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
                model.train()


def eval_during_training(args, accelerator, eval_dataloader, model):
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss
        losses.append(accelerator.gather(loss))

    losses = torch.cat(losses)
    total_loss = torch.mean(losses)

    return total_loss


if __name__ == "__main__":
    main()
