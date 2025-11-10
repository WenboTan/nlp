#!/usr/bin/env python3
"""
Full training script for A1 assignment. Designed to be run interactively or via SLURM.

Example:
  python3 train_full.py \
    --train_file /data/courses/2025_dat450_dit247/assignments/a1/train.txt \
    --val_file   /data/courses/2025_dat450_dit247/assignments/a1/val.txt \
    --epochs 1 --train_batch 4 --eval_batch 4 --lr 2e-3 \
    --embedding_size 128 --hidden_size 128 \
    --output_dir ./a1_model_small --save_tokenizer a1_tokenizer.pkl
"""

import argparse
import os
import random
import numpy as np
import torch
from torch.utils.data import Subset, DataLoader

from datasets import load_dataset

from A1_skeleton import (
    build_tokenizer,
    A1RNNModelConfig,
    A1RNNModel,
    A1Trainer,
    text_collate_fn,
    compute_validation_perplexity,
    predict_next_word,
)


def parse_args():
    p = argparse.ArgumentParser()
    # data
    p.add_argument('--train_file', type=str, required=True, help='Path to train.txt')
    p.add_argument('--val_file', type=str, required=True, help='Path to val.txt')
    p.add_argument('--save_tokenizer', type=str, default=None, help='If set, save tokenizer to this path')
    p.add_argument('--subsample', type=int, default=None,
                   help='If set, take first N examples from each split for quick debugging')
    # model / training
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--train_batch', type=int, default=4)
    p.add_argument('--eval_batch', type=int, default=4)
    p.add_argument('--lr', type=float, default=2e-3)
    p.add_argument('--max_voc_size', type=int, default=20000)
    p.add_argument('--model_max_length', type=int, default=128)
    p.add_argument('--embedding_size', type=int, default=128)
    p.add_argument('--hidden_size', type=int, default=128)
    p.add_argument('--output_dir', type=str, default='./a1_model_full')
    p.add_argument('--use_cpu', action='store_true', help='Force CPU even if CUDA available')
    p.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    p.add_argument('--dataloader_workers', type=int, default=4, help='(Hint) A1Trainer may ignore this; safe to keep')
    p.add_argument('--seed', type=int, default=2025)
    # eval/printing
    p.add_argument('--predict_prompt', type=str, default='She lives in San',
                   help='A short prompt for next-word prediction demo')
    p.add_argument('--topk', type=int, default=5, help='Top-K for next-word prediction')
    return p.parse_args()


class SimpleArgs:
    """Mimic a minimal TrainingArguments-like object for A1Trainer."""
    pass


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_text_datasets(train_file: str, val_file: str):
    """Load text datasets with HuggingFace and strip empty lines."""
    dataset = load_dataset('text', data_files={'train': train_file, 'val': val_file})
    dataset['train'] = dataset['train'].filter(lambda x: x['text'].strip() != '')
    dataset['val'] = dataset['val'].filter(lambda x: x['text'].strip() != '')
    return dataset


def maybe_subsample(dataset_dict, n: int):
    """Subsample first n samples from each split for quick debugging."""
    if n is None:
        return dataset_dict
    dataset_dict['train'] = Subset(dataset_dict['train'], range(min(n, len(dataset_dict['train']))))
    dataset_dict['val'] = Subset(dataset_dict['val'], range(min(n, len(dataset_dict['val']))))
    return dataset_dict


def main():
    args = parse_args()
    set_seed(args.seed)

    print('[Info] Building tokenizer from:', args.train_file)
    tokenizer = build_tokenizer(
        args.train_file,
        max_voc_size=args.max_voc_size,
        model_max_length=args.model_max_length
    )
    print('[Info] Vocab size =', len(tokenizer))
    if args.save_tokenizer:
        tokenizer.save(args.save_tokenizer)
        print(f'[Info] Tokenizer saved to {args.save_tokenizer}')

    print('[Info] Loading datasets...')
    ds = load_text_datasets(args.train_file, args.val_file)
    if args.subsample:
        print('[Info] Subsampling to', args.subsample)
        ds = maybe_subsample(ds, args.subsample)

    print('[Info] Building model...')
    cfg = A1RNNModelConfig(
        vocab_size=len(tokenizer),
        embedding_size=args.embedding_size,
        hidden_size=args.hidden_size
    )
    model = A1RNNModel(cfg)

    # Build minimal args object expected by A1Trainer
    ta = SimpleArgs()
    ta.optim = 'adamw_torch'
    ta.eval_strategy = 'epoch'
    ta.use_cpu = args.use_cpu
    ta.no_cuda = args.no_cuda
    ta.learning_rate = args.lr
    ta.num_train_epochs = args.epochs
    ta.per_device_train_batch_size = args.train_batch
    ta.per_device_eval_batch_size = args.eval_batch
    ta.output_dir = args.output_dir
    # Note: current A1Trainer in skeleton ignores num_workers/pin_memory; keeping here is harmless
    ta.dataloader_num_workers = args.dataloader_workers

    print('[Info] Starting training...')
    trainer = A1Trainer(model, ta, ds['train'], ds['val'], tokenizer)
    trainer.train()

    # ---- Final validation PPL (explicit printout) ----
    device = trainer.select_device()
    val_loader = DataLoader(
        ds['val'],
        batch_size=args.eval_batch,
        shuffle=False,
        collate_fn=text_collate_fn
    )
    nll, ppl = compute_validation_perplexity(model, tokenizer, val_loader, device)
    print(f"[FINAL] Validation NLL={nll:.4f}  PPL={ppl:.2f}")

    # ---- Next-word prediction demo ----
    top = predict_next_word(model, tokenizer, args.predict_prompt, topk=args.topk, device=device)
    print(f'Next-word prediction for: "{args.predict_prompt}"')
    for w, s in top:
        print(f"  {w:15s} logit={s:.3f}")


if __name__ == '__main__':
    main()
