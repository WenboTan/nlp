#!/usr/bin/env python3
"""
Full training script for A1 assignment. Designed to be run interactively or via SLURM.
Example (interactive):
  python3 train_full.py --epochs 3 --train_batch 64 --eval_batch 64 --lr 5e-4 --output_dir ./a1_model_full

This script builds tokenizer from train.txt, constructs the model from A1_skeleton.A1RNNModelConfig,
creates DataLoaders with the helper make_dataloader, and runs training using A1Trainer.
"""

import argparse
import os
from A1_skeleton import build_tokenizer, A1RNNModelConfig, A1RNNModel, A1Trainer, load_text_datasets, subsample_dataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--train_batch', type=int, default=64)
    p.add_argument('--eval_batch', type=int, default=64)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--max_voc_size', type=int, default=20000)
    p.add_argument('--model_max_length', type=int, default=128)
    p.add_argument('--embedding_size', type=int, default=256)
    p.add_argument('--hidden_size', type=int, default=512)
    p.add_argument('--output_dir', type=str, default='./a1_model_full')
    p.add_argument('--subsample', type=int, default=None, help='If set, subsample both splits to this many examples for quick debugging')
    p.add_argument('--use_cpu', action='store_true', help='Force CPU even if CUDA available')
    p.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    p.add_argument('--dataloader_workers', type=int, default=4)
    return p.parse_args()


class SimpleArgs:
    pass


def main():
    args_cmd = parse_args()

    print('Building tokenizer...')
    tokenizer = build_tokenizer('train.txt', max_voc_size=args_cmd.max_voc_size, model_max_length=args_cmd.model_max_length)
    print('Vocab size:', len(tokenizer))

    print('Loading datasets...')
    ds = load_text_datasets('train.txt', 'val.txt')
    if args_cmd.subsample is not None:
        print('Subsampling to', args_cmd.subsample)
        ds = subsample_dataset(ds, n=args_cmd.subsample)

    print('Building model...')
    cfg = A1RNNModelConfig(vocab_size=len(tokenizer), embedding_size=args_cmd.embedding_size, hidden_size=args_cmd.hidden_size)
    model = A1RNNModel(cfg)

    # Build args object expected by A1Trainer
    a = SimpleArgs()
    a.optim = 'adamw_torch'
    a.eval_strategy = 'epoch'
    a.use_cpu = args_cmd.use_cpu
    a.no_cuda = args_cmd.no_cuda
    a.learning_rate = args_cmd.lr
    a.num_train_epochs = args_cmd.epochs
    a.per_device_train_batch_size = args_cmd.train_batch
    a.per_device_eval_batch_size = args_cmd.eval_batch
    a.dataloader_num_workers = args_cmd.dataloader_workers
    a.output_dir = args_cmd.output_dir

    # Create trainer and run
    trainer = A1Trainer(model, a, ds['train'], ds['val'], tokenizer)
    trainer.train()


if __name__ == '__main__':
    main()
