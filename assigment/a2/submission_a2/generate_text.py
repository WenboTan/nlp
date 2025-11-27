#!/usr/bin/env python3
"""
Text generation script for A2 assignment.
Implements random sampling with temperature and top-K sampling.

Example:
  python3 generate_text.py \
    --model_dir ./a2_model \
    --tokenizer_file a2_tokenizer.pkl \
    --prompt "In natural language processing, a Transformer" \
    --max_length 50 --temperature 0.8 --topk 50
"""

import argparse
import torch
import sys

sys.path.append('/data/users/wenbota/nlp/assigment/1/a1')
from A1_skeleton import A1Tokenizer

from A2_skeleton import A2Transformer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_dir', type=str, required=True, help='Directory with saved model')
    p.add_argument('--tokenizer_file', type=str, required=True, help='Path to tokenizer pickle file')
    p.add_argument('--prompt', type=str, required=True, help='Text prompt to start generation')
    p.add_argument('--max_length', type=int, default=50, help='Maximum generation length (in tokens)')
    p.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature (lower = more conservative)')
    p.add_argument('--topk', type=int, default=None, help='Top-K sampling (keep only top K tokens)')
    p.add_argument('--use_cpu', action='store_true', help='Force CPU')
    p.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    return p.parse_args()


def select_device(use_cpu=False):
    if use_cpu:
        return torch.device('cpu')
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def generate_text(model, tokenizer, prompt, max_length=50, temperature=1.0, topk=None, device=None):
    """
    Generate text using random sampling with temperature and top-K.
    
    Args:
        model: The language model
        tokenizer: Tokenizer to encode/decode text
        prompt: Initial text prompt
        max_length: Maximum number of generation steps
        temperature: Controls randomness (higher = more random, lower = more conservative)
        topk: If set, only sample from top-K most probable tokens
        device: Device to run on
    
    Returns:
        Generated text as a string
    """
    model.eval()
    
    # Encode prompt
    enc = tokenizer(prompt, truncation=True, padding=False, return_tensors='pt')
    input_ids = enc['input_ids']
    
    if device is not None:
        input_ids = input_ids.to(device)
    
    eos_token_id = tokenizer.eos_token_id
    generated_tokens = input_ids[0].tolist()
    
    with torch.no_grad():
        for _ in range(max_length):
            # Get model predictions
            logits = model(input_ids)  # [1, seq_len, vocab_size]
            
            # Get logits for the last token
            next_token_logits = logits[0, -1, :]  # [vocab_size]
            
            # Apply temperature
            next_token_logits = next_token_logits / temperature
            
            # Apply top-K filtering
            if topk is not None and topk > 0:
                # Get top-K values and indices
                topk_values, topk_indices = torch.topk(next_token_logits, min(topk, next_token_logits.size(-1)))
                
                # Create a mask for top-K tokens
                mask = torch.full_like(next_token_logits, float('-inf'))
                mask[topk_indices] = topk_values
                next_token_logits = mask
            
            # Convert to probabilities
            probs = torch.softmax(next_token_logits, dim=-1)
            
            # Sample next token
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Check for EOS
            if next_token.item() == eos_token_id:
                break
            
            # Append to sequence
            generated_tokens.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
    
    # Decode generated tokens
    # Skip BOS and EOS tokens for cleaner output
    result_tokens = []
    for token_id in generated_tokens:
        if token_id not in [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id]:
            result_tokens.append(tokenizer.int_to_str.get(token_id, tokenizer.unk_token))
    
    return ' '.join(result_tokens)


def main():
    args = parse_args()
    
    if args.seed is not None:
        torch.manual_seed(args.seed)
    
    # Load tokenizer
    print(f'[Info] Loading tokenizer from: {args.tokenizer_file}')
    tokenizer = A1Tokenizer.from_file(args.tokenizer_file)
    print(f'[Info] Vocab size: {len(tokenizer)}')
    
    # Load model
    print(f'[Info] Loading model from: {args.model_dir}')
    model = A2Transformer.from_pretrained(args.model_dir)
    
    device = select_device(args.use_cpu)
    print(f'[Info] Device: {device}')
    model.to(device)
    
    # Generate text
    print(f'\n[Prompt] {args.prompt}')
    print(f'[Settings] max_length={args.max_length}, temperature={args.temperature}, topk={args.topk}')
    print('-' * 80)
    
    generated = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_length=args.max_length,
        temperature=args.temperature,
        topk=args.topk,
        device=device
    )
    
    print(generated)
    print('-' * 80)


if __name__ == '__main__':
    main()
