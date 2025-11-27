#!/usr/bin/env python3
"""
展示模型的next-word prediction能力
"""

import sys
sys.path.append('/data/users/wenbota/nlp/assigment/1/a1')
from A1_skeleton import A1Tokenizer

import torch
from A2_skeleton import A2Transformer


def predict_next_word(model, tokenizer, text, topk=5, device=None):
    """预测下一个词"""
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


# 加载模型和tokenizer
print("Loading model and tokenizer...")
tokenizer = A1Tokenizer.from_file('a2_tokenizer.pkl')
model = A2Transformer.from_pretrained('./a2_model_lr1e-4_b16_h256_l4')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f"Device: {device}\n")

# 测试不同的prompts
test_prompts = [
    "The capital of France is",
    "In machine learning, deep neural networks",
    "The president of the United States",
    "Python is a programming language used for",
    "Natural language processing deals with",
    "The weather in Stockholm is usually",
    "Scientists have discovered that",
]

print("="*70)
print("Next-Word Prediction Results")
print("="*70)

for prompt in test_prompts:
    print(f"\nPrompt: \"{prompt}\"")
    predictions = predict_next_word(model, tokenizer, prompt, topk=5, device=device)
    print("Top predictions:")
    for word, prob in predictions:
        print(f"  {word:20s} prob={prob:.3f}")

print("\n" + "="*70)
