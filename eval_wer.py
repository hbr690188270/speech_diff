import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)


import argparse
import json
import logging
import math
import random
import numpy as np
import time
import tqdm
from jiwer import wer

import datasets
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

from transformers import AutoTokenizer
from transformers import GPT2Tokenizer
from lightning.pytorch.callbacks import TQDMProgressBar

from src.model import SentenceLevelAE
from src.data_utils import MyDataCollator
from src.utils import move_to_target_device
from src.metric import token_level_accuracy, token_level_accuracy_v2
from src.args_util import parse_args

logger = logging.getLogger(__name__)
torch.set_printoptions(sci_mode=False)


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
    
    state_dict = torch.load(args.encoder_path)['state_dict']
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict = False)

    del state_dict
    torch.cuda.empty_cache()

    model.requires_grad_(False)
    model.eval()

    device = torch.device('cuda')
    model = model.to(device)


    pred_list = []
    orig_list = []
    # for step, batch in enumerate(train_dataloader):
    for batch in tqdm.tqdm(eval_dataloader):
        batch = move_to_target_device(batch, device)
        if not args.use_kl:
            sent_embeddings, raw_sent_embeddings = model.encode(batch)
            prefix = sent_embeddings
        else:
            posterior, raw_sent_embeddings = model.encode(batch)
            prefix = posterior.mode()

        with torch.no_grad():
            outputs = model.generate(
                sentence_embeddings = prefix
            )
        output_sentence = decoder_tokenizer.batch_decode(outputs, skip_special_tokens = True)
        pred_list += output_sentence

        orig_sentence = decoder_tokenizer.batch_decode(batch['decoder_input_ids'], skip_special_tokens = True)
        orig_list += orig_sentence

    wer_list = np.array([wer(orig_list[i], pred_list[i]) for i in range(len(pred_list))])
    print(np.mean(wer_list))


if __name__ == "__main__":
    main()



