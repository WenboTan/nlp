#!/usr/bin/env python3
"""
完整的文本生成演示 - 满足Assignment 2的所有要求
包括：
1. Next-word prediction
2. 完整文本生成（temperature + top-K sampling）
3. 不同参数的对比实验
"""

import sys
sys.path.append('/data/users/wenbota/nlp/assigment/1/a1')
from A1_skeleton import A1Tokenizer

import torch
from A2_skeleton import A2Transformer


def predict_next_word(model, tokenizer, text, topk=5, device=None):
    """预测下一个词（作业要求的第一部分）"""
    model.eval()
    enc = tokenizer(text, truncation=True, padding=False, return_tensors='pt')
    input_ids = enc['input_ids']
    
    if input_ids.size(1) < 1:
        return []
    
    if device is not None:
        input_ids = input_ids.to(device)
    
    with torch.no_grad():
        logits = model(input_ids)
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


def generate_text(model, tokenizer, prompt, max_length=50, temperature=1.0, topk=None, device=None):
    """
    完整文本生成（作业要求的第二部分）
    使用random sampling with temperature and top-K
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
            logits = model(input_ids)
            next_token_logits = logits[0, -1, :]
            
            # Apply temperature
            next_token_logits = next_token_logits / temperature
            
            # Apply top-K filtering
            if topk is not None and topk > 0:
                topk_values, topk_indices = torch.topk(next_token_logits, min(topk, next_token_logits.size(-1)))
                mask = torch.full_like(next_token_logits, float('-inf'))
                mask[topk_indices] = topk_values
                next_token_logits = mask
            
            # Convert to probabilities and sample
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Check for EOS
            if next_token.item() == eos_token_id:
                break
            
            # Append to sequence
            generated_tokens.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
    
    # Decode generated tokens
    result_tokens = []
    for token_id in generated_tokens:
        if token_id not in [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id]:
            result_tokens.append(tokenizer.int_to_str.get(token_id, tokenizer.unk_token))
    
    return ' '.join(result_tokens)


print("="*80)
print("Assignment 2 - Complete Text Generation Demo")
print("="*80)

# Load model and tokenizer
print("\n[1] Loading model and tokenizer...")
tokenizer = A1Tokenizer.from_file('a2_tokenizer.pkl')
model = A2Transformer.from_pretrained('./a2_model_lr1e-4_b16_h256_l4')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f"    Device: {device}")
print(f"    Vocab size: {len(tokenizer)}")

# Part 1: Next-word prediction
print("\n" + "="*80)
print("PART 1: Next-Word Prediction")
print("="*80)

test_prompts_nextword = [
    "The capital of France is",
    "In machine learning, deep neural networks",
    "Natural language processing deals with",
]

for prompt in test_prompts_nextword:
    print(f"\nPrompt: \"{prompt}\"")
    predictions = predict_next_word(model, tokenizer, prompt, topk=5, device=device)
    print("Top 5 predictions:")
    for word, prob in predictions:
        print(f"  {word:20s} prob={prob:.4f}")

# Part 2: Full text generation with different parameters
print("\n" + "="*80)
print("PART 2: Full Text Generation (Random Sampling)")
print("="*80)

generation_prompts = [
    "In natural language processing, a Transformer",
    "Is Stockholm the capital of Sweden? Answer yes or no. The answer is",
    "Write a Python program that reverses a list.",
]

# Test different temperature values
temperatures = [0.5, 0.8, 1.2]
topk_value = 50

print("\n--- Experiment 1: Different Temperatures (topk=50) ---")
for prompt in generation_prompts[:1]:  # Use first prompt
    print(f"\n{'='*80}")
    print(f"Prompt: \"{prompt}\"")
    print('='*80)
    for temp in temperatures:
        print(f"\n  Temperature={temp}:")
        generated = generate_text(model, tokenizer, prompt, max_length=40, 
                                 temperature=temp, topk=topk_value, device=device)
        print(f"  {generated}")

# Test different top-K values
print("\n\n--- Experiment 2: Different Top-K values (temperature=0.8) ---")
topk_values = [10, 50, None]
for prompt in generation_prompts[:1]:  # Use first prompt
    print(f"\n{'='*80}")
    print(f"Prompt: \"{prompt}\"")
    print('='*80)
    for topk_val in topk_values:
        topk_str = str(topk_val) if topk_val is not None else "None (full vocab)"
        print(f"\n  Top-K={topk_str}:")
        generated = generate_text(model, tokenizer, prompt, max_length=40, 
                                 temperature=0.8, topk=topk_val, device=device)
        print(f"  {generated}")

# Test all three suggested prompts
print("\n\n--- Experiment 3: All Three Suggested Prompts ---")
for i, prompt in enumerate(generation_prompts, 1):
    print(f"\n{'='*80}")
    print(f"Prompt {i}: \"{prompt}\"")
    print('='*80)
    generated = generate_text(model, tokenizer, prompt, max_length=50, 
                             temperature=0.8, topk=50, device=device)
    print(f"Generated: {generated}")

print("\n" + "="*80)
print("Generation Complete!")
print("="*80)
print("\nObservations:")
print("- Lower temperature (0.5): More conservative, repetitive")
print("- Higher temperature (1.2): More creative, potentially less coherent")
print("- Smaller top-K (10): Safer choices, more consistent")
print("- Larger top-K (50) or None: More diverse, potentially more surprising")
