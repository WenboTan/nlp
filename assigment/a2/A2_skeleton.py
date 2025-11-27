
import torch
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel, PretrainedConfig


class A2ModelConfig(PretrainedConfig):
    """Configuration object that stores hyperparameters that define the Transformer language model."""
    def __init__(self, vocab_size=None, hidden_size=None, intermediate_size=None, num_attention_heads=None, 
                 num_hidden_layers=None,
                 rope_theta=None, hidden_act='silu', max_position_embeddings=None, rms_norm_eps=None, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.num_attention_heads = num_attention_heads
        self.rope_theta = rope_theta
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers



class A2MLP(nn.Module):
    """The MLP layer of the Transformer. Uses the SwiGLU architecture."""
    def __init__(self, config):
        super().__init__()
        assert(config.hidden_act == 'silu')
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, hidden_states):
        # SwiGLU: gate_proj(x) * act_fn * up_proj(x), then project down
        gate = self.act_fn(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)
        intermediate = gate * up
        output = self.down_proj(intermediate)
        return output

# This is optional, since you can use PyTorch's RMSNorm.
class A2RMSNorm(nn.Module):
    """RMS layer normalization."""
    def __init__(self, config):
        super().__init__()
        self.eps = config.rms_norm_eps
        self.weight = nn.Parameter(torch.ones(config.hidden_size))

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)


class A2Attention(nn.Module):
    """The multi-head attention layer of the Transformer. Uses standard scaled dot-product attention with causal masking."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        
        assert self.hidden_size % self.num_attention_heads == 0, \
            "hidden_size must be divisible by num_attention_heads"
        
        # Query, key, value projections (W_q, W_k, W_v)
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        # Output projection (W_o)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        # Normalizers for query and key
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps, elementwise_affine=True)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps, elementwise_affine=True)

    def forward(self, hidden_states, rope_rotations):
        b, m, d = hidden_states.shape  # batch, seq_len, hidden_size
        n_h = self.num_attention_heads
        d_h = self.head_dim
        
        # Compute query, key, value representations
        q = self.q_proj(hidden_states)  # [b, m, d]
        k = self.k_proj(hidden_states)  # [b, m, d]
        v = self.v_proj(hidden_states)  # [b, m, d]
        
        # Reshape for multi-head attention: [b, m, d] -> [b, n_h, m, d_h]
        q = q.view(b, m, n_h, d_h).transpose(1, 2)
        k = k.view(b, m, n_h, d_h).transpose(1, 2)
        v = v.view(b, m, n_h, d_h).transpose(1, 2)
        
        # Apply layer normalization to query and key
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # Apply RoPE rotations to query and key
        q, k = apply_rotary_pos_emb(q, k, rope_rotations)
        
        # Compute attention using PyTorch's scaled_dot_product_attention
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True
        )
        
        # Combine attention heads: [b, n_h, m, d_h] -> [b, m, d]
        attn_out = attn_out.transpose(1, 2).reshape(b, m, d)
        
        # Apply output projection
        output = self.o_proj(attn_out)
        return output


class A2DecoderLayer(nn.Module):
    """A complete Transformer decoder layer."""
    def __init__(self, config):
        super().__init__()
        self.self_attn = A2Attention(config)
        self.mlp = A2MLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps, elementwise_affine=True)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps, elementwise_affine=True)

    def forward(self, hidden_states, rope_rotations):
        # Self-attention with residual connection
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, rope_rotations)
        hidden_states = residual + hidden_states
        
        # MLP with residual connection
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class A2Transformer(PreTrainedModel):
    """A language model based on the Transformer architecture."""
    
    config_class = A2ModelConfig

    def __init__(self, config):
        super().__init__(config)
        
        self.config = config
        self.vocab_size = config.vocab_size
        
        # Embedding layer
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Rotary embeddings
        self.rotary_emb = A2RotaryEmbedding(config)
        
        # Transformer decoder layers
        self.layers = nn.ModuleList([
            A2DecoderLayer(config) for _ in range(config.num_hidden_layers)
        ])
        
        # Final layer normalization
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps, elementwise_affine=True)
        
        # Unembedding layer (output projection to vocabulary)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # This line should be called after you have set up all components.
        self.post_init()


    def forward(self, input_ids):
        # Get RoPE rotations
        rope_rotations = self.rotary_emb(input_ids)
        
        # Embedding
        hidden_states = self.embed_tokens(input_ids)
        
        # Pass through all transformer decoder layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, rope_rotations)
        
        # Final normalization
        hidden_states = self.norm(hidden_states)
        
        # Unembedding (project to vocabulary)
        logits = self.lm_head(hidden_states)
        
        return logits


#### RoPE implementation (copied and simplified from HuggingFace). ####

def apply_rotary_pos_emb(q, k, rope_rotations, unsqueeze_dim=1):
    """Applies precomputed RoPE rotations to the query and key representations."""
    assert(q.shape == k.shape)
    assert(len(q.shape) == 4)
    cos, sin = rope_rotations
    assert(q.shape[2] == cos.shape[1])
    assert(q.shape[3] == cos.shape[2])    
    q_type, k_type = q.dtype, k.dtype
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(q_type), k_embed.to(k_type)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

class A2RotaryEmbedding(nn.Module):
    """RoPE position representation for use in Transformer attention."""

    def __init__(self, config, device=None):
        super().__init__()
        rope_theta = config.rope_theta
        head_dim = config.hidden_size // config.num_attention_heads
        partial_rotary_factor = 1.0
        dim = int(head_dim * partial_rotary_factor)
        self.inv_freq = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))

    @torch.no_grad()
    def forward(self, x):
        position_ids = torch.arange(0, x.shape[1], device=x.device).unsqueeze(0)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
            return cos, sin
