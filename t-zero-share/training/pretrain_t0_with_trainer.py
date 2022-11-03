# coding=utf-8

import sys
import logging
import os
import random
from dataclasses import dataclass, field
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
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    DataCollatorForSeq2Seq,
    AdamW,
    SchedulerType,
    get_scheduler,
    set_seed,
)
from transformers.optimization import Adafactor, AdafactorSchedule
from transformers.file_utils import PaddingStrategy
from templates import DatasetTemplates
from pretraining_utils import load_training_task_list, load_all_dataset

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "The model checkpoint for weights initialization. "
                          "Don't set if you want to train a model from scratch."})

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    train_task_list: Optional[str] = field(default=None, metadata={"help": "用于预训练的任务列表"})
    data_dir: Optional[str] = field(default=None, metadata={"help": "用于预训练的数据集列表（需要已经预处理过，每个prompt一个文件夹）"})

    train_file: Optional[str] = field(default=None, metadata={"help": "预训练文件，全部都处理好的"})
    eval_file: Optional[str] = field(default=None, metadata={"help": "预训练文件，全部都处理好的"})

    template_dir: Optional[str] = field(default=None, metadata={"help": "模板文件的路径"})

    max_seq_length: int = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
                    " sequences shorter will be padded if `--pad_to_max_lengh` is passed."})
    max_tgt_length: int = field(
        default=256,
        metadata={"help": "Target max length. Sequences longer than this will be truncated."})

    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."})

    # T0训练的时候包含eos，但是t0没有包含
    input_eos: bool = field(
        default=False,
        metadata={"help": "T0 was trained without EOS in its input sequences, which is the default in this script. "
                          "However, T5 was pretrained with EOS in its input sequences. See README for more info."})

    pad_to_max_length: bool = field(
        default=False,
        metadata={"help": "If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used."})


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 兼容没有trainer的实现
    data_args.max_length = data_args.max_seq_length
    data_args.target_max_length = data_args.max_tgt_length

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    train_task_list = load_training_task_list(data_args.train_task_list)

    raw_train_dataset, raw_eval_dataset = load_all_dataset(data_args, logger, train_task_list,
                                                           data_args.data_dir, data_args.template_dir)
    logger.info(f'raw_train_dataset: {raw_train_dataset}')
    logger.info(f'raw_eval_dataset: {raw_eval_dataset}')

    # 读模型
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_args.model_name_or_path, config=config)

    train_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if training_args.fp16 or training_args.bf16 else None
    )
    # train_dataloader = DataLoader(
    #     raw_train_dataset,
    #     shuffle=True,
    #     collate_fn=train_collator,
    #     batch_size=args.per_device_train_batch_size
    # )

    # # eval的collator
    # eval_collator = default_data_collator
    # eval_dataloader = DataLoader(raw_eval_dataset, collate_fn=eval_collator, batch_size=args.per_device_eval_batch_size)

    # Initialize our Trainer
    logger.info(f'初始化Trainer')

    if training_args.adafactor:
        logger.info(f'使用Adafactor优化器')
        # optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)

        optimizer = Adafactor(model.parameters(), scale_parameter=False, relative_step=False, warmup_init=False,
                              lr=training_args.learning_rate)
        logger.info(f'adafactor: {optimizer}')
        # lr_scheduler = AdafactorSchedule(optimizer)
        lr_scheduler = None
        trainer = Trainer(
            model=model,
            args=training_args,
            optimizers=(optimizer, lr_scheduler),
            train_dataset=raw_train_dataset,
            eval_dataset=raw_eval_dataset,
            tokenizer=tokenizer,
            data_collator=train_collator,
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=raw_train_dataset,
            eval_dataset=raw_eval_dataset,
            tokenizer=tokenizer,
            data_collator=train_collator,
        )

    # 准备开始训练
    checkpoint = None
    logger.info(f'现在开始训练')
    train_result = trainer.train()
    trainer.save_model()  # Saves the tokenizer too for easy upload
    metrics = train_result.metrics
    logger.info(f'debug train_result.metrics: {metrics}')
    metrics["train_samples"] = len(raw_train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info('Done!')


if __name__ == "__main__":
    main()
