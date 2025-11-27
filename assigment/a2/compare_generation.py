#!/usr/bin/env python3
"""
Compare text generation between your trained model and pre-trained OLMo-2.

Example:
  python3 compare_generation.py \
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
from transformers import AutoTokenizer, AutoModelForCausalLM


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_dir', type=str, required=True, help='Directory with your trained model')
    p.add_argument('--tokenizer_file', type=str, required=True, help='Path to your tokenizer pickle file')
    p.add_argument('--prompt', type=str, required=True, help='Text prompt to start generation')
    p.add_argument('--max_length', type=int, default=50, help='Maximum generation length')
    p.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    p.add_argument('--topk', type=int, default=50, help='Top-K sampling')
    p.add_argument('--use_cpu', action='store_true', help='Force CPU')
    p.add_argument('--seed', type=int, default=42, help='Random seed')
    return p.parse_args()


def select_device(use_cpu=False):
    if use_cpu:
        return torch.device('cpu')
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def generate_with_custom_model(model, tokenizer, prompt, max_length, temperature, topk, device):
    """Generate text with your custom model."""
    model.eval()
    enc = tokenizer(prompt, truncation=True, padding=False, return_tensors='pt')
    input_ids = enc['input_ids'].to(device)
    eos_token_id = tokenizer.eos_token_id
    generated_tokens = input_ids[0].tolist()
    
    with torch.no_grad():
        for _ in range(max_length):
            logits = model(input_ids)
            next_token_logits = logits[0, -1, :] / temperature
            
            if topk is not None and topk > 0:
                topk_values, topk_indices = torch.topk(next_token_logits, min(topk, next_token_logits.size(-1)))
                mask = torch.full_like(next_token_logits, float('-inf'))
                mask[topk_indices] = topk_values
                next_token_logits = mask
            
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            if next_token.item() == eos_token_id:
                break
            
            generated_tokens.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
    
    result_tokens = []
    for token_id in generated_tokens:
        if token_id not in [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id]:
            result_tokens.append(tokenizer.int_to_str.get(token_id, tokenizer.unk_token))
    
    return ' '.join(result_tokens)


def generate_with_pretrained(model, tokenizer, prompt, max_length, temperature, topk, device):
    """Generate text with pre-trained OLMo-2."""
    model.eval()
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    input_ids = inputs['input_ids']
    eos_token_id = tokenizer.eos_token_id
    
    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids)
            logits = outputs.logits  # Note: outputs is CausalLMOutputWithPast
            next_token_logits = logits[0, -1, :] / temperature
            
            if topk is not None and topk > 0:
                topk_values, topk_indices = torch.topk(next_token_logits, min(topk, next_token_logits.size(-1)))
                mask = torch.full_like(next_token_logits, float('-inf'))
                mask[topk_indices] = topk_values
                next_token_logits = mask
            
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            if next_token.item() == eos_token_id:
                break
            
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
    
    generated = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return generated


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    
    device = select_device(args.use_cpu)
    print(f'[Info] Device: {device}')
    
    # Load your model
    print(f'\n[Info] Loading your model from: {args.model_dir}')
    your_tokenizer = A1Tokenizer.from_file(args.tokenizer_file)
    your_model = A2Transformer.from_pretrained(args.model_dir)
    your_model.to(device)
    
    # Load pre-trained OLMo-2
    print(f'[Info] Loading pre-trained OLMo-2 model...')
    local_dir = '/data/courses/2025_dat450_dit247/models/OLMo-2-0425-1B'
    pretrained_tokenizer = AutoTokenizer.from_pretrained(local_dir, local_files_only=True)
    pretrained_model = AutoModelForCausalLM.from_pretrained(local_dir, local_files_only=True)
    pretrained_model.to(device)
    
    # Generate with both models
    print(f'\n{"="*80}')
    print(f'PROMPT: {args.prompt}')
    print(f'Settings: max_length={args.max_length}, temperature={args.temperature}, topk={args.topk}')
    print(f'{"="*80}')
    
    print(f'\n--- YOUR MODEL ---')
    your_output = generate_with_custom_model(
        your_model, your_tokenizer, args.prompt,
        args.max_length, args.temperature, args.topk, device
    )
    print(your_output)
    
    print(f'\n--- PRE-TRAINED OLMo-2 (1B) ---')
    pretrained_output = generate_with_pretrained(
        pretrained_model, pretrained_tokenizer, args.prompt,
        args.max_length, args.temperature, args.topk, device
    )
    print(pretrained_output)
    
    print(f'\n{"="*80}')


if __name__ == '__main__':
    main()
