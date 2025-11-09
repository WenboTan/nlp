#!/usr/bin/env python3
"""
Evaluation script for A1 assignment.
Generates the required outputs: perplexity on validation set and next-word predictions.
"""

import torch
import math
from A1_skeleton import A1RNNModel, build_tokenizer, load_text_datasets, make_dataloader

def calculate_perplexity(model, dataloader, tokenizer, device):
    """Calculate perplexity on a dataset."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    loss_func = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction='sum')
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids']
            X = input_ids[:, :-1].to(device)
            Y = input_ids[:, 1:].to(device)
            
            logits = model(X)  # (B, T-1, V)
            V = logits.size(-1)
            logits_flat = logits.reshape(-1, V)
            targets_flat = Y.reshape(-1)
            
            loss = loss_func(logits_flat, targets_flat)
            n_tokens = (targets_flat != tokenizer.pad_token_id).sum().item()
            
            total_loss += loss.item()
            total_tokens += n_tokens
    
    avg_loss = total_loss / max(1, total_tokens)
    perplexity = math.exp(avg_loss)
    return perplexity, avg_loss


def predict_next_word(model, tokenizer, input_text, device, top_k=5):
    """Predict the next word given input text."""
    model.eval()
    
    # Tokenize input
    tokens = tokenizer([input_text], truncation=True, padding=False, return_tensors='pt')
    input_ids = tokens['input_ids'].to(device)  # (1, T)
    
    with torch.no_grad():
        logits = model(input_ids)  # (1, T, V)
        next_token_logits = logits[0, -1, :]  # (V,)
        
        # Get top-k predictions
        probs = torch.softmax(next_token_logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, k=top_k)
        
        predictions = []
        for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
            word = tokenizer.int_to_str.get(idx, '<UNK>')
            predictions.append((word, prob))
    
    return predictions


def main():
    print("="*80)
    print("A1 Assignment - Evaluation Results")
    print("="*80)
    print()
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = build_tokenizer('train.txt', max_voc_size=20000, model_max_length=128)
    print(f"Vocabulary size: {len(tokenizer)}")
    print()
    
    # Load model
    print("Loading trained model...")
    model = A1RNNModel.from_pretrained('./a1_model_full')
    
    # Select device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    print()
    
    # Load validation dataset
    print("Loading validation dataset...")
    datasets = load_text_datasets('train.txt', 'val.txt')
    val_loader = make_dataloader(datasets['val'], tokenizer, batch_size=64, shuffle=False, num_workers=0)
    print()
    
    # Calculate perplexity on validation set
    print("-"*80)
    print("PERPLEXITY ON VALIDATION SET")
    print("-"*80)
    perplexity, avg_loss = calculate_perplexity(model, val_loader, tokenizer, device)
    print(f"Validation Loss (per token): {avg_loss:.6f}")
    print(f"Validation Perplexity: {perplexity:.4f}")
    print()
    
    # Next-word predictions
    print("-"*80)
    print("NEXT-WORD PREDICTIONS")
    print("-"*80)
    
    # Example prompts - you can customize these
    test_prompts = [
        "The president of the",
        "In the year",
        "The capital of",
        "Scientists have discovered",
        "The most important thing is",
    ]
    
    for prompt in test_prompts:
        print(f"\nInput: \"{prompt}\"")
        predictions = predict_next_word(model, tokenizer, prompt, device, top_k=5)
        print("Top 5 predictions:")
        for i, (word, prob) in enumerate(predictions, 1):
            print(f"  {i}. {word:20s} (probability: {prob:.4f})")
    
    print()
    print("="*80)
    print("Evaluation completed!")
    print("="*80)


if __name__ == '__main__':
    main()
