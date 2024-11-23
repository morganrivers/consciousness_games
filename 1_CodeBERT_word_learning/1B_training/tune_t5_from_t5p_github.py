
"""
Finetune CodeT5+ models on any Seq2Seq LM tasks
You can customize your own training data by following the HF dataset format to cache it to args.cache_data
Author: Yue Wang
Date: June 2023

Modified slightly by Morgan Rivers Sep 2024
Source: https://github.com/salesforce/CodeT5/blob/main/CodeT5%2B/tune_codet5p_seq2seq.py
        #ADDED FOR VALUE HEAD CONFIDENCE TRAINING (https://arxiv.org/pdf/2207.05221)
        # https://chatgpt.com/share/6740a5a5-c698-8006-9dce-f3cf5d29cd60

NOTE: Run with the command:
python tune_t5_from_t5p_github.py --epochs 5 --batch-size-per-replica 2 --grad-acc-steps 8 --fp16
"""
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn  # Ensure this import is present
from copy import deepcopy
from typing import Optional, Tuple, Union

import os
import csv
import pprint
import argparse
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, Trainer, TrainerCallback
import torch
from transformers import AdamW, get_linear_schedule_with_warmup
from trl import PPOTrainer, PPOConfig
from rl_utils import perform_rl_step
from trl.models.modeling_value_head import AutoModelForSeq2SeqLMWithValueHead

from transformers import TrainerCallback
import re
from transformers import T5ForConditionalGeneration #, Seq2SeqLMOutput
from transformers.modeling_outputs import Seq2SeqLMOutput

import torch
import torch.nn.functional as F
import math

class LogToCSVCallback(TrainerCallback):
    def __init__(self, csv_path):
        self.csv_path = csv_path
        # Initialize CSV with headers if the file doesn't exist
        if not os.path.exists(csv_path):
            with open(self.csv_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['step', 'training_loss', 'eval_loss'])

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        # Get training and evaluation losses, if available
        training_loss = logs.get('loss', None)  # Training loss
        eval_loss = logs.get('eval_loss', None)  # Evaluation loss

        # Log values to CSV
        if training_loss is not None or eval_loss is not None:
            with open(self.csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([state.global_step, training_loss, eval_loss])

def load_tokenize_data(args):
    train_data_dir = os.path.join(args.cache_data, 'train')
    test_data_dir = os.path.join(args.cache_data, 'test')
    if os.path.exists(train_data_dir) and os.path.exists(test_data_dir):
        # Load the datasets from disk
        train_data = load_from_disk(train_data_dir)
        test_data = load_from_disk(test_data_dir)
        print(f'  ==> Loaded {len(train_data)} training samples from {train_data_dir}')
        print(f'  ==> Loaded {len(test_data)} test samples from {test_data_dir}')
        
        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.load)
        
        # Preprocess the data
        def preprocess_function(examples):
            # Tokenize the input and target texts
            inputs = examples['input']
            targets = examples['target']
            
            model_inputs = tokenizer(inputs, max_length=args.max_source_len, padding="max_length", truncation=True)
            labels = tokenizer(targets, max_length=args.max_target_len, padding="max_length", truncation=True)
            
            # Replace pad_token_id with -100 in labels
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs
        
        # Map the preprocessing function to the datasets
        train_data = train_data.map(
            preprocess_function,
            batched=True,
            remove_columns=train_data.column_names,
            num_proc=4,
            load_from_cache_file=False,
        )
        test_data = test_data.map(
            preprocess_function,
            batched=True,
            remove_columns=test_data.column_names,
            num_proc=4,
            load_from_cache_file=False,
        )
        print(f'  ==> Tokenized {len(train_data)} training samples')
        print(f'  ==> Tokenized {len(test_data)} test samples')

        # Use half of the test data as eval_data
        from datasets import Dataset
        eval_data = test_data.shuffle(seed=42).select(range(len(test_data)//2))

        return train_data, eval_data
    else:
        print(f'No cached data found at {args.cache_data}. Please ensure that your data is saved there.')
        exit(1)

# Modify the main function to accept eval_data
def main(args):
    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    # Save command to file
    with open(os.path.join(args.save_dir, "command.txt"), 'w') as f:
        f.write(pprint.pformat(argsdict))

    # Load and tokenize data using the tokenizer from `args.load`
    train_data, eval_data = load_tokenize_data(args)

    if args.data_num != -1:
        train_data = train_data.select([i for i in range(args.data_num)])

    model = AutoModelForSeq2SeqLM.from_pretrained(args.load)
    print(f"  ==> Loaded model from {args.load}, model size {model.num_parameters()}")

    run_training(args, model, train_data, eval_data)

# Modify run_training to accept eval_data
def run_training(args, model, train_data, eval_data):
    print(f"Starting main loop")

    training_args = TrainingArguments(
        report_to='tensorboard',
        output_dir=args.save_dir,
        overwrite_output_dir=False,

        do_train=True,
        save_strategy='epoch',

        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size_per_replica,
        gradient_accumulation_steps=args.grad_acc_steps,

        learning_rate=args.lr,
        weight_decay=0.05,
        warmup_steps=args.lr_warmup_steps,

        logging_dir=args.save_dir,
        logging_first_step=True,
        logging_steps=args.log_freq,
        save_total_limit=1,

        dataloader_drop_last=True,
        dataloader_num_workers=4,

        local_rank=args.local_rank,
        deepspeed=args.deepspeed,
        fp16=args.fp16,

        # **Add these lines to set evaluation strategy**
        evaluation_strategy='steps',
        eval_steps=3,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        #tokenizer=tokenizer  # Pass the tokenizer for decoding model outputs
    )

    trainer.train()

    if args.local_rank in [0, -1]:
        final_checkpoint_dir = os.path.join(args.save_dir, "final_checkpoint")
        model.save_pretrained(final_checkpoint_dir)
        print(f'  ==> Finish training and save to {final_checkpoint_dir}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CodeT5+ finetuning on Seq2Seq LM task")
    parser.add_argument('--data-num', default=-1, type=int)
    parser.add_argument('--cache-data', default='processed_dataset', type=str)
    parser.add_argument('--max-source-len', default=512, type=int)
    parser.add_argument('--max-target-len', default=512, type=int)
    parser.add_argument('--load', default='Salesforce/codet5p-220m', type=str)

    # Training
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--lr-warmup-steps', default=200, type=int)
    parser.add_argument('--batch-size-per-replica', default=1, type=int)
    parser.add_argument('--grad-acc-steps', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--deepspeed', default=None, type=str)
    parser.add_argument('--fp16', default=False, action='store_true')

    # Logging and stuff
    parser.add_argument('--save-dir', default="saved_models/summarize_python", type=str)
    parser.add_argument('--log-freq', default=10, type=int)
    parser.add_argument('--save-freq', default=500, type=int)
    parser.add_argument('--eval-steps', default=100, type=int, help='Evaluate the model every N steps')

    # PPO stuff
    parser.add_argument('--rl-step-interval', default=100, type=int, help='Interval (in steps) to perform RL step')
    parser.add_argument('--rl-batch-size', default=1, type=int, help='Batch size for RL step')
    parser.add_argument('--ppo-epochs', default=4, type=int, help='Number of PPO epochs')
    parser.add_argument('--ppo-lr', default=1.41e-5, type=float, help='Learning rate for PPO')
    parser.add_argument('--rl-grad-acc-steps', default=1, type=int, help='Gradient accumulation steps')  # Ensure non-zero
    parser.add_argument('--mini-batch-size', default=None, type=int, help='Mini-batch size for PPO (optional)')

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    main(args)
