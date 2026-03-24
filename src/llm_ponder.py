"""
LLM PonderNet: Adaptive Computation Metacognition for Language Models
======================================================================

This extends PonderNet to work with transformer-based language models.

Key Ideas:
1. Early Exit: Don't use all layers if confident
2. Iterative Refinement: Refine hidden states before final prediction
3. Calibrated Confidence: Computation time encodes uncertainty

Three Variants:
- LayerPonder: Exit early from transformer layers
- TokenPonder: Multiple refinement steps per token
- ChainPonder: Iterative chain-of-thought with halting

Author: Metacognition Research
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass


@dataclass
class LLMPonderConfig:
    """Configuration for LLM PonderNet"""
    vocab_size: int = 32000
    hidden_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    max_seq_len: int = 512
    dropout: float = 0.1
    
    # PonderNet specific
    max_ponder_steps: int = 8
    ponder_mode: str = "layer"  # "layer", "token", or "chain"
    lambda_p: float = 0.05     # KL regularization
    prior_p: float = 0.3       # Geometric prior parameter
    
    # Early exit threshold (for inference)
    confidence_threshold: float = 0.9


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding for better position encoding"""
    
    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len
        
    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(q: torch.Tensor, k: torch.Tensor, 
                     cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class PonderAttention(nn.Module):
    """Multi-head attention with halting prediction"""
    
    def __init__(self, config: LLMPonderConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads
        
        self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        
        self.rotary = RotaryEmbedding(self.head_dim, config.max_seq_len)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape
        
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary embeddings
        cos, sin = self.rotary(x, T)
        q, k = apply_rotary_emb(q, k, cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0))
        
        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.o_proj(out)


class PonderFFN(nn.Module):
    """Feed-forward network with SwiGLU activation"""
    
    def __init__(self, config: LLMPonderConfig):
        super().__init__()
        hidden = int(config.hidden_dim * 8/3)
        self.w1 = nn.Linear(config.hidden_dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, config.hidden_dim, bias=False)
        self.w3 = nn.Linear(config.hidden_dim, hidden, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class PonderTransformerBlock(nn.Module):
    """Transformer block with halting mechanism"""
    
    def __init__(self, config: LLMPonderConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        
        self.attn = PonderAttention(config)
        self.ffn = PonderFFN(config)
        self.norm1 = nn.RMSNorm(config.hidden_dim)
        self.norm2 = nn.RMSNorm(config.hidden_dim)
        
        # Halting head for this layer
        self.halter = nn.Sequential(
            nn.Linear(config.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Pre-norm architecture
        h = x + self.attn(self.norm1(x), mask)
        h = h + self.ffn(self.norm2(h))
        
        # Compute halting probability (pool over sequence)
        halt_logits = self.halter(h.mean(dim=1))  # [B, 1]
        halt_prob = halt_logits.squeeze(-1)  # [B]
        
        return h, halt_prob


class LayerPonderLLM(nn.Module):
    """
    LLM with layer-wise early exit (PonderNet style).
    
    Key Innovation: Instead of using all N layers, the model learns
    to exit early when confident. Computation time (layers used) 
    encodes uncertainty.
    """
    
    def __init__(self, config: LLMPonderConfig):
        super().__init__()
        self.config = config
        
        # Token embedding
        self.embed = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.embed_dropout = nn.Dropout(config.dropout)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            PonderTransformerBlock(config, i) for i in range(config.num_layers)
        ])
        
        # Final norm and output
        self.norm = nn.RMSNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        
        # Tie embeddings
        self.lm_head.weight = self.embed.weight
        
        self._init_weights()
        
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                
    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        B, T = input_ids.shape
        device = input_ids.device
        
        # Create causal mask
        if attention_mask is None:
            attention_mask = torch.ones(B, T, device=device)
        causal_mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
        causal_mask = ~causal_mask  # True = attend, False = mask
        
        # Embed
        h = self.embed_dropout(self.embed(input_ids))
        
        # Process through layers with halting
        all_logits = []
        all_halts = []
        
        for layer in self.layers:
            h, halt_prob = layer(h, causal_mask)
            
            # Compute logits at this layer
            logits = self.lm_head(self.norm(h))
            
            all_logits.append(logits)
            all_halts.append(halt_prob)
        
        # Stack: [B, num_layers, T, V] and [B, num_layers]
        all_logits = torch.stack(all_logits, dim=1)
        all_halts = torch.stack(all_halts, dim=1)
        
        # Compute halting distribution (geometric-like)
        halt_dist = self._compute_halt_distribution(all_halts)
        
        # Weighted combination of predictions from each layer
        # [B, num_layers, 1, 1] * [B, num_layers, T, V] -> [B, T, V]
        weighted_logits = (halt_dist.unsqueeze(-1).unsqueeze(-1) * all_logits).sum(dim=1)
        
        # Expected layers used
        layer_indices = torch.arange(1, self.config.num_layers + 1, device=device).float()
        expected_layers = (halt_dist * layer_indices).sum(dim=1)
        
        # Compute confidence
        probs = F.softmax(weighted_logits, dim=-1)
        softmax_conf = probs.max(dim=-1)[0].mean(dim=-1)  # Average over sequence
        comp_uncertainty = expected_layers / self.config.num_layers
        confidence = 0.7 * softmax_conf + 0.3 * (1 - comp_uncertainty)
        
        outputs = {
            'logits': weighted_logits,
            'probs': probs,
            'confidence': confidence,
            'expected_layers': expected_layers,
            'halt_dist': halt_dist,
            'all_halts': all_halts,
        }
        
        if labels is not None:
            outputs['loss'] = self._compute_loss(outputs, labels)
            
        return outputs
    
    def _compute_halt_distribution(self, halts: torch.Tensor) -> torch.Tensor:
        """Convert halt probabilities to halting distribution"""
        B = halts.shape[0]
        device = halts.device
        
        log_not_halt = torch.log(1 - halts + 1e-8)
        cumsum_log = torch.cumsum(log_not_halt, dim=1)
        cumsum_shifted = torch.cat([
            torch.zeros(B, 1, device=device),
            cumsum_log[:, :-1]
        ], dim=1)
        
        halt_dist = halts * torch.exp(cumsum_shifted)
        
        # Ensure sums to 1
        remainder = 1 - halt_dist.sum(dim=1, keepdim=True)
        halt_dist = torch.cat([
            halt_dist[:, :-1], 
            halt_dist[:, -1:] + remainder.clamp(min=0)
        ], dim=1)
        
        return halt_dist
    
    def _compute_loss(self, outputs: Dict, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute PonderNet loss: task + KL regularization"""
        # Task loss (language modeling)
        logits = outputs['logits']
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        task_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100
        )
        
        # KL divergence from geometric prior
        halt_dist = outputs['halt_dist']
        num_layers = halt_dist.shape[1]
        
        prior_dist = torch.zeros(num_layers, device=halt_dist.device)
        for t in range(num_layers):
            prior_dist[t] = self.config.prior_p * ((1 - self.config.prior_p) ** t)
        prior_dist = prior_dist / prior_dist.sum()
        
        kl_loss = F.kl_div(
            torch.log(halt_dist + 1e-8),
            prior_dist.unsqueeze(0).expand_as(halt_dist),
            reduction='batchmean'
        )
        
        total_loss = task_loss + self.config.lambda_p * kl_loss
        
        return {
            'total': total_loss,
            'task': task_loss,
            'kl': kl_loss
        }
    
    def generate(self, input_ids: torch.Tensor, 
                 max_new_tokens: int = 100,
                 temperature: float = 1.0,
                 early_exit: bool = True) -> Tuple[torch.Tensor, Dict]:
        """Generate with optional early exit for efficiency"""
        self.eval()
        generated = input_ids.clone()
        all_confidences = []
        all_layers_used = []
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = self.forward(generated)
                
                # Get next token logits
                next_logits = outputs['logits'][:, -1, :] / temperature
                next_probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(next_probs, num_samples=1)
                
                generated = torch.cat([generated, next_token], dim=1)
                all_confidences.append(outputs['confidence'].item())
                all_layers_used.append(outputs['expected_layers'].item())
                
                # Early stopping if very confident (e.g., EOS)
                if early_exit and outputs['confidence'].item() > 0.99:
                    break
                    
        return generated, {
            'confidences': all_confidences,
            'layers_used': all_layers_used,
            'avg_layers': sum(all_layers_used) / len(all_layers_used)
        }


class TokenPonderLLM(nn.Module):
    """
    LLM with token-level pondering.
    
    For each token position, the model can "think" for multiple steps
    before committing to a prediction.
    """
    
    def __init__(self, config: LLMPonderConfig):
        super().__init__()
        self.config = config
        
        # Embedding
        self.embed = nn.Embedding(config.vocab_size, config.hidden_dim)
        
        # Single transformer layer for pondering
        self.ponder_block = PonderTransformerBlock(config, 0)
        
        # Refinement GRU (like original PonderNet)
        self.refine_gru = nn.GRUCell(config.hidden_dim, config.hidden_dim)
        
        # Per-step halter
        self.step_halter = nn.Sequential(
            nn.Linear(config.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Output
        self.norm = nn.RMSNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight
        
    def forward(self, input_ids: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        B, T = input_ids.shape
        device = input_ids.device
        
        # Initial embedding
        h = self.embed(input_ids)
        
        # Process through transformer once
        h, _ = self.ponder_block(h)
        
        # Now ponder at each position
        all_logits = []
        all_halts = []
        
        for step in range(self.config.max_ponder_steps):
            # Refine each position
            h_flat = h.view(B * T, -1)
            h_refined = self.refine_gru(h_flat, h_flat)
            h = h_refined.view(B, T, -1)
            
            # Compute outputs at this step
            logits = self.lm_head(self.norm(h))
            halt_prob = self.step_halter(h.mean(dim=1)).squeeze(-1)
            
            all_logits.append(logits)
            all_halts.append(halt_prob)
        
        # Stack and compute halt distribution
        all_logits = torch.stack(all_logits, dim=1)  # [B, steps, T, V]
        all_halts = torch.stack(all_halts, dim=1)    # [B, steps]
        
        halt_dist = self._compute_halt_distribution(all_halts)
        
        # Weighted logits
        weighted_logits = (halt_dist.unsqueeze(-1).unsqueeze(-1) * all_logits).sum(dim=1)
        
        # Confidence from computation
        step_indices = torch.arange(1, self.config.max_ponder_steps + 1, device=device).float()
        expected_steps = (halt_dist * step_indices).sum(dim=1)
        
        probs = F.softmax(weighted_logits, dim=-1)
        softmax_conf = probs.max(dim=-1)[0].mean(dim=-1)
        comp_uncertainty = expected_steps / self.config.max_ponder_steps
        confidence = 0.7 * softmax_conf + 0.3 * (1 - comp_uncertainty)
        
        outputs = {
            'logits': weighted_logits,
            'confidence': confidence,
            'expected_steps': expected_steps,
            'halt_dist': halt_dist,
        }
        
        if labels is not None:
            outputs['loss'] = self._compute_loss(outputs, labels)
            
        return outputs
    
    def _compute_halt_distribution(self, halts: torch.Tensor) -> torch.Tensor:
        B = halts.shape[0]
        device = halts.device
        
        log_not_halt = torch.log(1 - halts + 1e-8)
        cumsum_log = torch.cumsum(log_not_halt, dim=1)
        cumsum_shifted = torch.cat([
            torch.zeros(B, 1, device=device),
            cumsum_log[:, :-1]
        ], dim=1)
        
        halt_dist = halts * torch.exp(cumsum_shifted)
        remainder = 1 - halt_dist.sum(dim=1, keepdim=True)
        halt_dist = torch.cat([halt_dist[:, :-1], halt_dist[:, -1:] + remainder.clamp(min=0)], dim=1)
        
        return halt_dist
    
    def _compute_loss(self, outputs: Dict, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = outputs['logits']
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        task_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100
        )
        
        halt_dist = outputs['halt_dist']
        num_steps = halt_dist.shape[1]
        
        prior_dist = torch.zeros(num_steps, device=halt_dist.device)
        for t in range(num_steps):
            prior_dist[t] = self.config.prior_p * ((1 - self.config.prior_p) ** t)
        prior_dist = prior_dist / prior_dist.sum()
        
        kl_loss = F.kl_div(
            torch.log(halt_dist + 1e-8),
            prior_dist.unsqueeze(0).expand_as(halt_dist),
            reduction='batchmean'
        )
        
        return {
            'total': task_loss + self.config.lambda_p * kl_loss,
            'task': task_loss,
            'kl': kl_loss
        }


# =============================================================================
# EXPERIMENT: Test on Small Scale
# =============================================================================

def run_llm_ponder_experiment():
    """Test LLM PonderNet on synthetic language modeling task"""
    print("=" * 70)
    print("LLM PONDERNET EXPERIMENT")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    # Small config for testing
    config = LLMPonderConfig(
        vocab_size=1000,
        hidden_dim=256,
        num_layers=6,
        num_heads=4,
        max_seq_len=64,
        max_ponder_steps=6,
    )
    
    print(f"\nConfig: {config.num_layers} layers, {config.hidden_dim} dim")
    print(f"PonderNet: max {config.max_ponder_steps} ponder steps\n")
    
    # Synthetic data: random sequences
    train_data = torch.randint(0, config.vocab_size, (100, 32))
    
    # Create model
    model = LayerPonderLLM(config)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")
    
    # Training
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    print("\n--- Training ---")
    model.train()
    for epoch in range(10):
        total_loss = 0
        for i in range(0, len(train_data), 8):
            batch = train_data[i:i+8]
            
            optimizer.zero_grad()
            outputs = model(batch, labels=batch)
            loss = outputs['loss']['total']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 2 == 0:
            avg_loss = total_loss / (len(train_data) // 8)
            print(f"Epoch {epoch+1}: loss={avg_loss:.4f}")
    
    # Evaluation
    print("\n--- Evaluation ---")
    model.eval()
    with torch.no_grad():
        test_batch = train_data[:8]
        outputs = model(test_batch)
        
        print(f"Confidence: {outputs['confidence'].mean():.3f}")
        print(f"Expected layers: {outputs['expected_layers'].mean():.2f} / {config.num_layers}")
        print(f"Halt distribution: {outputs['halt_dist'][0].cpu().numpy().round(3)}")
    
    # Test generation
    print("\n--- Generation Test ---")
    prompt = torch.randint(0, config.vocab_size, (1, 5))
    generated, gen_stats = model.generate(prompt, max_new_tokens=10)
    
    print(f"Generated {generated.shape[1] - 5} new tokens")
    print(f"Average layers used: {gen_stats['avg_layers']:.2f}")
    print(f"Confidence range: [{min(gen_stats['confidences']):.3f}, {max(gen_stats['confidences']):.3f}]")
    
    return model, config


if __name__ == "__main__":
    run_llm_ponder_experiment()
