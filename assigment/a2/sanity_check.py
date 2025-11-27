#!/usr/bin/env python3
"""
Sanity checks for A2 Transformer components.
Tests each component to ensure correct shapes and no crashes.
"""

import torch
from A2_skeleton import (
    A2ModelConfig,
    A2MLP,
    A2RMSNorm,
    A2Attention,
    A2DecoderLayer,
    A2Transformer
)


def test_mlp():
    print("=" * 60)
    print("Testing MLP Layer")
    print("=" * 60)
    
    config = A2ModelConfig(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=1024,
        num_attention_heads=4,
        num_hidden_layers=2,
        max_position_embeddings=128,
        rope_theta=10000.0,
        rms_norm_eps=1e-5
    )
    
    mlp = A2MLP(config)
    
    # Test with 3D tensor: [batch, seq_len, hidden_size]
    batch_size = 2
    seq_len = 10
    x = torch.randn(batch_size, seq_len, config.hidden_size)
    
    output = mlp(x)
    
    assert output.shape == x.shape, f"Expected shape {x.shape}, got {output.shape}"
    print(f"✓ Input shape:  {x.shape}")
    print(f"✓ Output shape: {output.shape}")
    print("✓ MLP test passed!\n")


def test_rmsnorm():
    print("=" * 60)
    print("Testing RMSNorm Layer")
    print("=" * 60)
    
    config = A2ModelConfig(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=1024,
        num_attention_heads=4,
        num_hidden_layers=2,
        max_position_embeddings=128,
        rope_theta=10000.0,
        rms_norm_eps=1e-5
    )
    
    # Test custom implementation
    norm1 = A2RMSNorm(config)
    
    # Test PyTorch built-in
    norm2 = torch.nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps, elementwise_affine=True)
    
    batch_size = 2
    seq_len = 10
    x = torch.randn(batch_size, seq_len, config.hidden_size)
    
    output1 = norm1(x)
    output2 = norm2(x)
    
    assert output1.shape == x.shape, f"Expected shape {x.shape}, got {output1.shape}"
    assert output2.shape == x.shape, f"Expected shape {x.shape}, got {output2.shape}"
    
    print(f"✓ Input shape:  {x.shape}")
    print(f"✓ Custom RMSNorm output shape: {output1.shape}")
    print(f"✓ PyTorch RMSNorm output shape: {output2.shape}")
    print("✓ RMSNorm test passed!\n")


def test_attention():
    print("=" * 60)
    print("Testing Multi-Head Attention Layer")
    print("=" * 60)
    
    config = A2ModelConfig(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=1024,
        num_attention_heads=4,
        num_hidden_layers=2,
        max_position_embeddings=128,
        rope_theta=10000.0,
        rms_norm_eps=1e-5
    )
    
    attention = A2Attention(config)
    
    batch_size = 2
    seq_len = 10
    x = torch.randn(batch_size, seq_len, config.hidden_size)
    
    # Create dummy input_ids for RoPE
    from A2_skeleton import A2RotaryEmbedding
    rotary_emb = A2RotaryEmbedding(config)
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    rope_rotations = rotary_emb(input_ids)
    
    output = attention(x, rope_rotations)
    
    assert output.shape == x.shape, f"Expected shape {x.shape}, got {output.shape}"
    print(f"✓ Input shape:  {x.shape}")
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Number of heads: {config.num_attention_heads}")
    print(f"✓ Head dimension: {config.hidden_size // config.num_attention_heads}")
    print("✓ Attention test passed!\n")


def test_decoder_layer():
    print("=" * 60)
    print("Testing Transformer Decoder Layer")
    print("=" * 60)
    
    config = A2ModelConfig(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=1024,
        num_attention_heads=4,
        num_hidden_layers=2,
        max_position_embeddings=128,
        rope_theta=10000.0,
        rms_norm_eps=1e-5
    )
    
    decoder_layer = A2DecoderLayer(config)
    
    batch_size = 2
    seq_len = 10
    x = torch.randn(batch_size, seq_len, config.hidden_size)
    
    # Create RoPE rotations
    from A2_skeleton import A2RotaryEmbedding
    rotary_emb = A2RotaryEmbedding(config)
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    rope_rotations = rotary_emb(input_ids)
    
    output = decoder_layer(x, rope_rotations)
    
    assert output.shape == x.shape, f"Expected shape {x.shape}, got {output.shape}"
    print(f"✓ Input shape:  {x.shape}")
    print(f"✓ Output shape: {output.shape}")
    print("✓ Decoder layer test passed!\n")


def test_full_transformer():
    print("=" * 60)
    print("Testing Full Transformer Model")
    print("=" * 60)
    
    config = A2ModelConfig(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=1024,
        num_attention_heads=4,
        num_hidden_layers=2,
        max_position_embeddings=128,
        rope_theta=10000.0,
        rms_norm_eps=1e-5
    )
    
    model = A2Transformer(config)
    
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    output = model(input_ids)
    
    expected_shape = (batch_size, seq_len, config.vocab_size)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
    
    print(f"✓ Input shape:  {input_ids.shape} (token IDs)")
    print(f"✓ Output shape: {output.shape} (logits)")
    print(f"✓ Vocab size: {config.vocab_size}")
    print(f"✓ Hidden size: {config.hidden_size}")
    print(f"✓ Number of layers: {config.num_hidden_layers}")
    print(f"✓ Number of heads: {config.num_attention_heads}")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Total parameters: {num_params:,}")
    print("✓ Full Transformer test passed!\n")


def test_forward_backward():
    print("=" * 60)
    print("Testing Forward and Backward Pass")
    print("=" * 60)
    
    config = A2ModelConfig(
        vocab_size=1000,
        hidden_size=128,
        intermediate_size=512,
        num_attention_heads=4,
        num_hidden_layers=2,
        max_position_embeddings=128,
        rope_theta=10000.0,
        rms_norm_eps=1e-5
    )
    
    model = A2Transformer(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    batch_size = 4
    seq_len = 20
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    # Forward pass
    logits = model(input_ids[:, :-1])  # [B, T-1, V]
    targets = input_ids[:, 1:]  # [B, T-1]
    
    # Compute loss
    B, T, V = logits.shape
    loss = loss_fn(logits.reshape(B * T, V), targets.reshape(B * T))
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"✓ Forward pass successful")
    print(f"✓ Loss computed: {loss.item():.4f}")
    print(f"✓ Backward pass successful")
    print(f"✓ Optimizer step successful")
    print("✓ Training loop test passed!\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Running A2 Transformer Sanity Checks")
    print("=" * 60 + "\n")
    
    try:
        test_mlp()
        test_rmsnorm()
        test_attention()
        test_decoder_layer()
        test_full_transformer()
        test_forward_backward()
        
        print("=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("✗ TEST FAILED!")
        print("=" * 60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
