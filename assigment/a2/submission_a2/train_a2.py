#!/usr/bin/env python3
"""
Training script for A2 assignment - Transformer Language Model.
Reuses tokenization and training utilities from A1, but uses the Transformer model.

Example:
  python3 train_a2.py \
    --train_file /data/courses/2025_dat450_dit247/assignments/a1/train.txt \
    --val_file   /data/courses/2025_dat450_dit247/assignments/a1/val.txt \
    --epochs 5 --train_batch 16 --eval_batch 16 --lr 1e-4 \
    --hidden_size 256 --num_layers 4 --num_heads 4 \
    --output_dir ./a2_model --save_tokenizer a2_tokenizer.pkl
"""

import argparse
import os
import random
import numpy as np
import torch
from torch.utils.data import Subset, DataLoader

from datasets import load_dataset

# Import tokenizer and utilities from A1
import sys
sys.path.append('/data/users/wenbota/nlp/assigment/1/a1')
from A1_skeleton import (
    build_tokenizer,
    text_collate_fn,
)

# Import Transformer model from A2
from A2_skeleton import A2ModelConfig, A2Transformer


def parse_args():
    p = argparse.ArgumentParser()
    # data
    p.add_argument('--train_file', type=str, required=True, help='Path to train.txt')
    p.add_argument('--val_file', type=str, required=True, help='Path to val.txt')
    p.add_argument('--save_tokenizer', type=str, default=None, help='If set, save tokenizer to this path')
    p.add_argument('--load_tokenizer', type=str, default=None, help='If set, load tokenizer from this path')
    p.add_argument('--subsample', type=int, default=None,
                   help='If set, take first N examples from each split for quick debugging')
    # model / training
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--train_batch', type=int, default=16)
    p.add_argument('--eval_batch', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--max_voc_size', type=int, default=20000)
    p.add_argument('--model_max_length', type=int, default=128)
    p.add_argument('--hidden_size', type=int, default=256)
    p.add_argument('--intermediate_size', type=int, default=None, help='MLP intermediate size (default: 4*hidden_size)')
    p.add_argument('--num_layers', type=int, default=4, help='Number of Transformer layers')
    p.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    p.add_argument('--output_dir', type=str, default='./a2_model')
    p.add_argument('--use_cpu', action='store_true', help='Force CPU even if CUDA available')
    p.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    p.add_argument('--seed', type=int, default=2025)
    # eval/printing
    p.add_argument('--predict_prompt', type=str, default='She lives in San',
                   help='A short prompt for next-word prediction demo')
    p.add_argument('--topk', type=int, default=5, help='Top-K for next-word prediction')
    return p.parse_args()


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


def select_device(use_cpu=False, no_cuda=False):
    if use_cpu:
        return torch.device('cpu')
    if not no_cuda and torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def train_epoch(model, tokenizer, dataloader, optimizer, loss_fn, device):
    """Train for one epoch."""
    model.train()
    running_tokens = 0
    running_loss_weighted = 0.0
    pad_id = tokenizer.pad_token_id
    
    for batch_texts in dataloader:
        enc = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            return_tensors='pt'
        )
        input_ids = enc['input_ids'].to(device)
        
        if input_ids.size(1) < 2:
            continue
        
        # Autoregressive: predict token t from context up to t-1
        X = input_ids[:, :-1].contiguous()
        Y = input_ids[:, 1:].contiguous()
        
        logits = model(X)
        B, Tm1, V = logits.shape
        
        loss = loss_fn(
            logits.reshape(B * Tm1, V),
            Y.reshape(B * Tm1)
        )
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            valid_mask = (Y != pad_id)
            n_valid = int(valid_mask.sum().item())
            running_tokens += n_valid
            running_loss_weighted += loss.item() * n_valid
    
    train_nll = running_loss_weighted / max(1, running_tokens)
    train_ppl = float(np.exp(train_nll))
    return train_nll, train_ppl


def evaluate(model, tokenizer, dataloader, loss_fn, device):
    """Evaluate on validation set."""
    model.eval()
    val_tokens = 0
    val_loss_weighted = 0.0
    pad_id = tokenizer.pad_token_id
    
    with torch.no_grad():
        for batch_texts in dataloader:
            enc = tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                return_tensors='pt'
            )
            input_ids = enc['input_ids'].to(device)
            
            if input_ids.size(1) < 2:
                continue
            
            X = input_ids[:, :-1].contiguous()
            Y = input_ids[:, 1:].contiguous()
            
            logits = model(X)
            B, Tm1, V = logits.shape
            
            loss = loss_fn(
                logits.reshape(B * Tm1, V),
                Y.reshape(B * Tm1)
            )
            
            valid_mask = (Y != pad_id)
            n_valid = int(valid_mask.sum().item())
            val_tokens += n_valid
            val_loss_weighted += loss.item() * n_valid
    
    val_nll = val_loss_weighted / max(1, val_tokens)
    val_ppl = float(np.exp(val_nll))
    return val_nll, val_ppl


def predict_next_word(model, tokenizer, text, topk=5, device=None):
    """
    Given a text prompt, return top-k next-token candidates (word, probability).
    """
    model.eval()
    enc = tokenizer(text, truncation=True, padding=False, return_tensors='pt')
    input_ids = enc['input_ids']
    
    if input_ids.size(1) < 1:
        return []
    
    if device is not None:
        input_ids = input_ids.to(device)
    
    with torch.no_grad():
        logits = model(input_ids)
        # Use -2 position (last real token before EOS) for better predictions
        if logits.size(1) >= 2:
            last_logits = logits[0, -2, :]
        else:
            last_logits = logits[0, -1, :]
        
        probs = torch.softmax(last_logits, dim=-1)
        scores, idxs = torch.topk(probs, k=topk)
    
    idxs = idxs.tolist()
    scores = scores.tolist()
    words = [tokenizer.int_to_str[i] for i in idxs]
    return list(zip(words, scores))


def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Build or load tokenizer
    if args.load_tokenizer and os.path.exists(args.load_tokenizer):
        print(f'[Info] Loading tokenizer from: {args.load_tokenizer}')
        from A1_skeleton import A1Tokenizer
        tokenizer = A1Tokenizer.from_file(args.load_tokenizer)
    else:
        print('[Info] Building tokenizer from:', args.train_file)
        tokenizer = build_tokenizer(
            args.train_file,
            max_voc_size=args.max_voc_size,
            model_max_length=args.model_max_length
        )
        if args.save_tokenizer:
            tokenizer.save(args.save_tokenizer)
            print(f'[Info] Tokenizer saved to {args.save_tokenizer}')
    
    print('[Info] Vocab size =', len(tokenizer))
    
    # Load datasets
    print('[Info] Loading datasets...')
    ds = load_text_datasets(args.train_file, args.val_file)
    if args.subsample:
        print('[Info] Subsampling to', args.subsample)
        ds = maybe_subsample(ds, args.subsample)
    
    # Build model
    print('[Info] Building Transformer model...')
    intermediate_size = args.intermediate_size if args.intermediate_size else 4 * args.hidden_size
    
    config = A2ModelConfig(
        vocab_size=len(tokenizer),
        hidden_size=args.hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=args.num_heads,
        num_hidden_layers=args.num_layers,
        max_position_embeddings=args.model_max_length,
        rope_theta=10000.0,
        rms_norm_eps=1e-5,
        hidden_act='silu'
    )
    model = A2Transformer(config)
    
    device = select_device(args.use_cpu, args.no_cuda)
    print('[Info] Device:', device)
    model.to(device)
    
    # Print model info
    num_params = sum(p.numel() for p in model.parameters())
    print(f'[Info] Model parameters: {num_params:,}')
    
    # Setup training
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    train_loader = DataLoader(
        ds['train'],
        batch_size=args.train_batch,
        shuffle=True,
        collate_fn=text_collate_fn
    )
    val_loader = DataLoader(
        ds['val'],
        batch_size=args.eval_batch,
        shuffle=False,
        collate_fn=text_collate_fn
    )
    
    # Training loop
    print('[Info] Starting training...')
    import time
    for epoch in range(args.epochs):
        t0 = time.time()
        train_nll, train_ppl = train_epoch(model, tokenizer, train_loader, optimizer, loss_fn, device)
        t1 = time.time()
        
        print(f"[Epoch {epoch+1}/{args.epochs}] "
              f"Train NLL={train_nll:.4f} PPL={train_ppl:.2f}  ({t1-t0:.1f}s)")
        
        # Validation
        val_nll, val_ppl = evaluate(model, tokenizer, val_loader, loss_fn, device)
        print(f"           Valid NLL={val_nll:.4f} PPL={val_ppl:.2f}")
    
    # Save model
    print(f"[Info] Saving to {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir, safe_serialization=False)
    config.save_pretrained(args.output_dir)
    
    # Final validation
    val_nll, val_ppl = evaluate(model, tokenizer, val_loader, loss_fn, device)
    print(f"[FINAL] Validation NLL={val_nll:.4f}  PPL={val_ppl:.2f}")
    
    # Next-word prediction demo
    top = predict_next_word(model, tokenizer, args.predict_prompt, topk=args.topk, device=device)
    print(f'\nNext-word prediction for: "{args.predict_prompt}"')
    for w, s in top:
        print(f"  {w:15s} prob={s:.3f}")


if __name__ == '__main__':
    main()
