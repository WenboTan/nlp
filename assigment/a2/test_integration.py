#!/usr/bin/env python3
"""
Quick test to verify A2 works with A1 tokenizer on a small dataset.
"""

import torch
import sys
import os

sys.path.append('/data/users/wenbota/nlp/assigment/1/a1')
from A1_skeleton import build_tokenizer

from A2_skeleton import A2ModelConfig, A2Transformer


def main():
    print("Testing A2 Transformer with A1 Tokenizer Integration")
    print("=" * 60)
    
    # Check if train file exists
    train_file = "/data/courses/2025_dat450_dit247/assignments/a1/train.txt"
    if not os.path.exists(train_file):
        print(f"Warning: {train_file} not found. Using dummy tokenizer.")
        # Create a dummy tokenizer for testing
        from A1_skeleton import A1Tokenizer
        str_to_int = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3}
        for i, word in enumerate(['the', 'a', 'is', 'in', 'and', 'of'], start=4):
            str_to_int[word] = i
        int_to_str = {v: k for k, v in str_to_int.items()}
        
        tokenizer = A1Tokenizer(
            str_to_int=str_to_int,
            int_to_str=int_to_str,
            pad_token='<PAD>',
            unk_token='<UNK>',
            bos_token='<BOS>',
            eos_token='<EOS>',
            pad_token_id=0,
            unk_token_id=1,
            bos_token_id=2,
            eos_token_id=3,
            tokenize_fun=lambda x: x.lower().split(),
            model_max_length=128
        )
        vocab_size = len(tokenizer)
    else:
        print("Building tokenizer from training file (first 100 lines)...")
        # Read first 100 lines for quick test
        with open(train_file, 'r') as f:
            lines = [f.readline() for _ in range(100)]
        
        # Create temp file
        temp_file = '/tmp/test_train.txt'
        with open(temp_file, 'w') as f:
            f.writelines(lines)
        
        tokenizer = build_tokenizer(
            temp_file,
            max_voc_size=1000,
            model_max_length=128
        )
        vocab_size = len(tokenizer)
        os.remove(temp_file)
    
    print(f"✓ Tokenizer created with vocab size: {vocab_size}")
    
    # Create small model
    config = A2ModelConfig(
        vocab_size=vocab_size,
        hidden_size=128,
        intermediate_size=512,
        num_attention_heads=4,
        num_hidden_layers=2,
        max_position_embeddings=128,
        rope_theta=10000.0,
        rms_norm_eps=1e-5
    )
    
    model = A2Transformer(config)
    print(f"✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test tokenization and forward pass
    test_texts = [
        "The quick brown fox jumps over the lazy dog",
        "In natural language processing"
    ]
    
    print("\nTesting forward pass with sample texts...")
    enc = tokenizer(test_texts, truncation=True, padding=True, return_tensors='pt')
    input_ids = enc['input_ids']
    print(f"✓ Input IDs shape: {input_ids.shape}")
    
    with torch.no_grad():
        logits = model(input_ids)
    print(f"✓ Output logits shape: {logits.shape}")
    print(f"✓ Expected vocab size in last dim: {vocab_size}")
    
    # Test training step
    print("\nTesting training step...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    X = input_ids[:, :-1]
    Y = input_ids[:, 1:]
    
    logits = model(X)
    B, T, V = logits.shape
    loss = loss_fn(logits.reshape(B * T, V), Y.reshape(B * T))
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"✓ Loss: {loss.item():.4f}")
    print(f"✓ Training step completed successfully")
    
    print("\n" + "=" * 60)
    print("✓ ALL INTEGRATION TESTS PASSED!")
    print("=" * 60)
    print("\nYou can now run the full training with:")
    print("  python3 train_a2.py --train_file <path> --val_file <path> ...")


if __name__ == "__main__":
    main()
