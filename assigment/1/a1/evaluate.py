#!/usr/bin/env python3
"""
Evaluation script for A1 assignment.
Prints: validation perplexity and next-word predictions.
"""

import argparse
import math
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset

from A1_skeleton import (
    A1RNNModel,
    A1Tokenizer,         # for .from_file
    build_tokenizer,     # fallback if no saved tokenizer
    text_collate_fn,
)

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--train_file', type=str, required=True, help='Path to train.txt')
    p.add_argument('--val_file', type=str, required=True, help='Path to val.txt')
    p.add_argument('--model_dir', type=str, required=True, help='Directory with saved model (from trainer_output)')
    p.add_argument('--tokenizer_file', type=str, default='a1_tokenizer.pkl', help='Path to saved tokenizer pickle')
    p.add_argument('--eval_batch', type=int, default=64)
    p.add_argument('--model_max_length', type=int, default=128, help='Used only if we must rebuild tokenizer')
    p.add_argument('--topk', type=int, default=5)
    p.add_argument('--prompt', type=str, default='She lives in San')
    return p.parse_args()


# ---------- Core eval helpers ----------
def calculate_perplexity(model, dataloader, tokenizer, device):
    """Calculate perplexity over a dataset that yields raw texts."""
    model.eval()
    pad_id = tokenizer.pad_token_id
    loss_func = torch.nn.CrossEntropyLoss(ignore_index=pad_id, reduction='sum')

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch_texts in dataloader:
            # Tokenize here
            enc = tokenizer(batch_texts, truncation=True, padding=True, return_tensors='pt')
            input_ids = enc['input_ids'].to(device)  # [B, T]
            if input_ids.size(1) < 2:
                continue

            X = input_ids[:, :-1]  # [B, T-1]
            Y = input_ids[:, 1:]   # [B, T-1]

            logits = model(X)      # [B, T-1, V]
            B, Tm1, V = logits.shape

            loss = loss_func(logits.reshape(B * Tm1, V), Y.reshape(B * Tm1))

            # Count valid tokens (non-PAD) in targets
            n_tokens = (Y != pad_id).sum().item()
            total_loss += loss.item()
            total_tokens += n_tokens

    avg_loss = total_loss / max(1, total_tokens)
    perplexity = math.exp(avg_loss)
    return perplexity, avg_loss


def predict_next_word(model, tokenizer, input_text, device, top_k=5):
    """Predict the next word given input text."""
    model.eval()
    with torch.no_grad():
        enc = tokenizer(input_text, truncation=True, padding=False, return_tensors='pt')
        input_ids = enc['input_ids'].to(device)  # [1, T]
        logits = model(input_ids)                # [1, T, V]
        next_token_logits = logits[0, -1, :]     # [V]
        # top-k from logits (equivalent to softmax+topk for ranking)
        top_scores, top_indices = torch.topk(next_token_logits, k=top_k)
        words = [tokenizer.int_to_str[int(i)] for i in top_indices]
        return list(zip(words, top_scores.tolist()))


# ---------- Main ----------
def main():
    args = parse_args()

    print("=" * 80)
    print("A1 Assignment - Evaluation")
    print("=" * 80)

    # Load tokenizer (prefer saved one to ensure vocab matches training)
    tokenizer = None
    try:
        tokenizer = A1Tokenizer.from_file(args.tokenizer_file)
        print(f"[Info] Loaded tokenizer from: {args.tokenizer_file} (vocab={len(tokenizer)})")
    except Exception as e:
        print(f"[Warn] Failed to load tokenizer from {args.tokenizer_file}: {e}")
        print("[Info] Rebuilding tokenizer from train_file (may mismatch training vocab).")
        tokenizer = build_tokenizer(args.train_file, max_voc_size=None, model_max_length=args.model_max_length)
        print(f"[Info] Built tokenizer (vocab={len(tokenizer)})")

    # Load model
    print(f"[Info] Loading model from: {args.model_dir}")
    model = A1RNNModel.from_pretrained(args.model_dir)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"[Info] Device: {device}")

    # Load validation dataset (raw text)
    dataset = load_dataset('text', data_files={'val': args.val_file})
    dataset['val'] = dataset['val'].filter(lambda x: x['text'].strip() != '')

    val_loader = DataLoader(
        dataset['val'],
        batch_size=args.eval_batch,
        shuffle=False,
        collate_fn=text_collate_fn
    )

    # Perplexity
    print("-" * 80)
    print("PERPLEXITY ON VALIDATION SET")
    print("-" * 80)
    ppl, avg_loss = calculate_perplexity(model, val_loader, tokenizer, device)
    print(f"Validation Loss (per token): {avg_loss:.6f}")
    print(f"Validation Perplexity: {ppl:.4f}")

    # Next-word predictions
    print("-" * 80)
    print("NEXT-WORD PREDICTIONS")
    print("-" * 80)
    prompt = args.prompt
    preds = predict_next_word(model, tokenizer, prompt, device, top_k=args.topk)
    print(f'\nInput: "{prompt}"')
    print("Top predictions:")
    for i, (w, s) in enumerate(preds, 1):
        print(f"  {i}. {w:20s} (logit: {s:.3f})")

    print("\n" + "=" * 80)
    print("Evaluation completed.")
    print("=" * 80)


if __name__ == '__main__':
    main()
