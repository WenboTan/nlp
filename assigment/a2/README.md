# Assignment 2 - Transformer Language Model

This directory contains the implementation of a Transformer-based language model following the OLMo 2 architecture.

## Files

- **A2_skeleton.py**: Complete Transformer implementation with:
  - `A2MLP`: SwiGLU MLP layer
  - `A2RMSNorm`: Root Mean Square Layer Normalization
  - `A2Attention`: Multi-head attention with RoPE positional embeddings
  - `A2DecoderLayer`: Complete decoder layer with residual connections
  - `A2Transformer`: Full Transformer language model
  - `A2RotaryEmbedding`: RoPE implementation (from HuggingFace)

- **train_a2.py**: Training script that reuses tokenizer and data utilities from A1
- **generate_text.py**: Text generation with temperature and top-K sampling
- **compare_generation.py**: Compare your model with pre-trained OLMo-2
- **sanity_check.py**: Test suite for all components
- **run_a2_slurm.sh**: SLURM batch script for training on GPU

## Quick Start

### 1. Run Sanity Checks

```bash
python3 sanity_check.py
```

This will test all Transformer components to ensure correct shapes and no crashes.

### 2. Train the Model

**Interactive (CPU/GPU):**
```bash
python3 train_a2.py \
    --train_file /data/courses/2025_dat450_dit247/assignments/a1/train.txt \
    --val_file /data/courses/2025_dat450_dit247/assignments/a1/val.txt \
    --save_tokenizer a2_tokenizer.pkl \
    --output_dir ./a2_model \
    --epochs 5 \
    --train_batch 16 \
    --eval_batch 16 \
    --lr 1e-4 \
    --hidden_size 256 \
    --num_layers 4 \
    --num_heads 4
```

**SLURM (GPU):**
```bash
sbatch run_a2_slurm.sh
```

### 3. Generate Text

```bash
python3 generate_text.py \
    --model_dir ./a2_model \
    --tokenizer_file a2_tokenizer.pkl \
    --prompt "In natural language processing, a Transformer" \
    --max_length 50 \
    --temperature 0.8 \
    --topk 50
```

### 4. Compare with Pre-trained Model

```bash
python3 compare_generation.py \
    --model_dir ./a2_model \
    --tokenizer_file a2_tokenizer.pkl \
    --prompt "In natural language processing, a Transformer" \
    --max_length 50 \
    --temperature 0.8 \
    --topk 50
```

## Model Architecture

The implementation follows the OLMo 2 architecture:

- **Embedding Layer**: Token embeddings
- **Transformer Layers**: Multiple decoder layers with:
  - Multi-head attention (with RoPE positional embeddings)
  - SwiGLU MLP
  - RMSNorm for normalization
  - Residual connections
- **Output Layer**: Unembedding to vocabulary logits

## Hyperparameters

Key hyperparameters (adjust in training script):

- `--hidden_size`: Hidden dimension (default: 256)
- `--num_layers`: Number of Transformer layers (default: 4)
- `--num_heads`: Number of attention heads (default: 4)
- `--intermediate_size`: MLP intermediate size (default: 4 * hidden_size)
- `--lr`: Learning rate (default: 1e-4)
- `--train_batch`: Training batch size (default: 16)
- `--epochs`: Number of training epochs (default: 5)
- `--max_voc_size`: Vocabulary size (default: 20000)
- `--model_max_length`: Maximum sequence length (default: 128)

## Text Generation Parameters

- `--temperature`: Controls randomness (lower = more conservative, default: 1.0)
- `--topk`: Top-K sampling (only sample from top K tokens, default: None)
- `--max_length`: Maximum generation length in tokens (default: 50)

## Example Prompts

Try these prompts for text generation:

1. `"In natural language processing, a Transformer"`
2. `"Is Stockholm the capital of Sweden? Answer yes or no. The answer is"`
3. `"Write a Python program that reverses a list."`

## Notes

- All `nn.Linear` layers use `bias=False` to match OLMo 2 architecture
- Uses RoPE (Rotary Position Embeddings) instead of absolute positional embeddings
- Implements causal (autoregressive) attention masking
- Compatible with HuggingFace's PreTrainedModel interface

## Requirements

- torch
- transformers
- datasets
- nltk (for tokenization from A1)
- numpy
