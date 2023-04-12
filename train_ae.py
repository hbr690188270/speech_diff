import os
import argparse
import json
import logging
import math
import random
import numpy as np
import time

import datasets
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from transformers import AutoTokenizer
from transformers import GPT2Tokenizer
import lightning as L
from lightning.pytorch import loggers
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.callbacks import TQDMProgressBar

from src.model import SentenceLevelAE
from src.data_utils import MyDataCollator 
from src.metric import token_level_accuracy, token_level_accuracy_v2
from src.args_util import parse_args

logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch`.

    Args:
        seed (`int`):
            The seed to set.
        device_specific (`bool`, *optional*, defaults to `False`):
            Whether to differ the seed on each device slightly with `self.process_index`.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class MyTQDMBar(TQDMProgressBar):
    def __init__(self, refresh_rate: int = 1, process_position: int = 0):
        super().__init__(refresh_rate, process_position)
    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch and trainer.is_global_zero:
            print()
        super().on_train_epoch_start(trainer, pl_module)

def main():
    args = parse_args()
    
    torch.autograd.set_detect_anomaly(True)
    torch.set_float32_matmul_precision('high')
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)


    raw_datasets = datasets.load_dataset("text", data_files = {'train': os.path.join(args.data_dir, 'train.txt'), 
                                                                'validation': os.path.join(args.data_dir, 'valid.txt')},
                                                cache_dir = os.path.join(os.path.curdir, 'dataset_cache'))

    print(raw_datasets)
    print(f"save to {os.path.join(os.path.curdir, 'dataset_cache')}")


    encoder_cache_dir = os.path.join(args.model_cache_dir, "diffcse")
    encoder_name = "voidism/diffcse-roberta-base-sts"
    encoder_tokenizer = AutoTokenizer.from_pretrained(encoder_name, cache_dir = encoder_cache_dir)

    decoder_cache_dir = os.path.join(args.model_cache_dir, "gpt2")
    decoder_name = "gpt2"

    decoder_tokenizer = GPT2Tokenizer.from_pretrained(decoder_name, cache_dir = decoder_cache_dir,
        use_fast=False, add_prefix_space = False)
    decoder_tokenizer.add_special_tokens({'pad_token': '<|pad|>'})

    # model = torch.compile(model)

    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]
    column_to_rm = [text_column_name]


    def tokenize_function(examples):
        encoder_inputs = encoder_tokenizer(
            examples[text_column_name],
            padding=False,
            truncation=True,
            max_length=50
        )
        decoder_inputs = decoder_tokenizer(
            examples[text_column_name],
            padding=False,
            truncation=True,
            max_length=50,
            add_special_tokens = True,
        )

        examples['input_ids'] = encoder_inputs['input_ids']
        examples['attention_mask'] = encoder_inputs['attention_mask']
        examples['decoder_input_ids'] = decoder_inputs['input_ids']
        return examples

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=column_to_rm,
        load_from_cache_file=True,
        desc="Running tokenizer on dataset line_by_line",
    )
    print(tokenized_datasets)
    print(text_column_name, column_to_rm)

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]

    # Conditional for small test subsets
    if len(train_dataset) > 3:
        # Log a few random samples from the training set:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    data_collator = MyDataCollator(encoder_tokenizer, decoder_tokenizer)

    train_sampler = None
    valid_sampler = None
    train_shuffle = True

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset, shuffle=train_shuffle, collate_fn=data_collator, batch_size=args.per_device_train_batch_size, sampler = train_sampler,
        prefetch_factor = 10, num_workers = 5
    )
    eval_dataloader = DataLoader(
        eval_dataset, shuffle=False, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size, sampler = valid_sampler,
        prefetch_factor = 10, num_workers = 5
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    if args.num_gpu > 1:
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / (args.gradient_accumulation_steps * args.num_gpu))
    else:
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)

    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch


    model = SentenceLevelAE(encoder_name = encoder_name, encoder_cache_dir = encoder_cache_dir,
                            decoder_name = decoder_name, decoder_cache_dir = decoder_cache_dir,
                            freeze_decoder = not args.train_decoder,
                            learning_rate = args.learning_rate,
                            weight_decay = args.weight_decay,
                            lr_scheduler_type = args.lr_scheduler_type, 
                            warmup_steps = args.num_warmup_steps,
                            num_training_steps = max_train_steps,
                            use_kl = args.use_kl,
                            kl_weight = args.kl_weight,
                            kl_scheduler = args.kl_scheduler,
                            cpg = args.cpg,
                            pretrained_path = args.encoder_path,
                            )

    if args.seed is not None:
        print("setting seed before training...")
        set_seed(args.seed)

    if args.num_gpu != len(args.gpu_ids) and args.num_gpu != 0:
        args.num_gpu = len(args.gpu_ids)
    if args.num_gpu == 0:
        devices = 'auto'
        accelerator = 'cpu'
        strategy = 'auto'
    elif args.num_gpu == 1:
        devices = args.gpu_ids
        accelerator = 'gpu'
        strategy = 'auto'
    else:
        accelerator="gpu"
        strategy="ddp_find_unused_parameters_true"
        # strategy="ddp"
        devices=args.gpu_ids

    checkpoint_callback = ModelCheckpoint(dirpath=args.output_dir, save_top_k=5, monitor="valid_loss",)
    lr_monitor = LearningRateMonitor(logging_interval='step')

    prog_bar = MyTQDMBar()

    all_callbacks = [checkpoint_callback, lr_monitor, prog_bar]

    if args.report_to == 'wandb':
        mylogger=loggers.WandbLogger(name = args.run_name, project = 'pl_diffusion', save_dir = './wandb/'),
    else:
        mylogger=False
    
    val_check_interval = 1.0

    trainer = L.Trainer(
            devices=devices,
            accelerator=accelerator,
            strategy=strategy,
            logger=mylogger,
            # val_check_interval = int(args.checkpointing_steps),
            val_check_interval = val_check_interval,
            max_epochs = args.num_train_epochs,
            accumulate_grad_batches = args.gradient_accumulation_steps,
            callbacks = all_callbacks,
            limit_val_batches = 1.0,
            gradient_clip_val = args.max_grad_norm,
            )
    trainer.fit(model, train_dataloader, eval_dataloader)


if __name__ == "__main__":
    main()



