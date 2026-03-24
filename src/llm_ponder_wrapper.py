"""
PonderNet Wrapper for Pre-trained LLMs
======================================

This module wraps existing pre-trained models (GPT-2, LLaMA, etc.) 
with PonderNet-style adaptive computation.

Instead of training from scratch, we:
1. Take a pre-trained model
2. Add lightweight halting heads to each layer
3. Fine-tune only the halting heads
4. Use the learned halting for calibrated confidence

This is practical for real-world deployment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass
import numpy as np

# Try to import transformers
try:
    from transformers import (
        AutoModelForCausalLM, 
        AutoTokenizer,
        GPT2LMHeadModel,
        GPT2Config
    )
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: transformers not installed. Install with: pip install transformers")


@dataclass
class PonderWrapperConfig:
    """Configuration for PonderNet wrapper"""
    max_ponder_steps: int = 8
    lambda_p: float = 0.05
    prior_p: float = 0.3
    freeze_base: bool = True  # Freeze base model, only train halters
    confidence_threshold: float = 0.9


class HaltingHead(nn.Module):
    """Lightweight halting prediction head"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Pool over sequence dimension
        pooled = hidden_states.mean(dim=1)  # [B, D]
        return self.net(pooled).squeeze(-1)  # [B]


class PonderGPT2(nn.Module):
    """
    GPT-2 wrapped with PonderNet-style adaptive computation.
    
    Key Features:
    - Uses pre-trained GPT-2 weights
    - Adds halting heads to each layer
    - Only halting heads are trained
    - Confidence derived from computation (layer exit)
    """
    
    def __init__(self, model_name: str = "gpt2", config: Optional[PonderWrapperConfig] = None):
        super().__init__()
        
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers library required")
        
        self.config = config or PonderWrapperConfig()
        
        # Load pre-trained model
        self.gpt2 = GPT2LMHeadModel.from_pretrained(model_name)
        self.hidden_dim = self.gpt2.config.n_embd
        self.num_layers = self.gpt2.config.n_layer
        
        # Freeze base model if specified
        if self.config.freeze_base:
            for param in self.gpt2.parameters():
                param.requires_grad = False
        
        # Add halting heads for each layer
        self.halting_heads = nn.ModuleList([
            HaltingHead(self.hidden_dim) for _ in range(self.num_layers)
        ])
        
        print(f"PonderGPT2: {self.num_layers} layers, {self.hidden_dim} dim")
        print(f"Trainable params: {sum(p.numel() for p in self.halting_heads.parameters()):,}")
        
    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        B = input_ids.shape[0]
        device = input_ids.device
        
        # Get embeddings
        inputs_embeds = self.gpt2.transformer.wte(input_ids)
        position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
        position_embeds = self.gpt2.transformer.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.gpt2.transformer.drop(hidden_states)
        
        # Process through layers, collecting outputs and halt probabilities
        all_logits = []
        all_halts = []
        
        for i, block in enumerate(self.gpt2.transformer.h):
            # Forward through transformer block
            outputs = block(hidden_states, attention_mask=attention_mask)
            hidden_states = outputs[0]
            
            # Compute logits at this layer
            normed = self.gpt2.transformer.ln_f(hidden_states)
            logits = self.gpt2.lm_head(normed)
            
            # Compute halting probability
            halt_prob = self.halting_heads[i](hidden_states)
            
            all_logits.append(logits)
            all_halts.append(halt_prob)
        
        # Stack
        all_logits = torch.stack(all_logits, dim=1)  # [B, L, T, V]
        all_halts = torch.stack(all_halts, dim=1)    # [B, L]
        
        # Compute halting distribution
        halt_dist = self._compute_halt_distribution(all_halts)
        
        # Weighted combination
        weighted_logits = (halt_dist.unsqueeze(-1).unsqueeze(-1) * all_logits).sum(dim=1)
        
        # Expected layers and confidence
        layer_indices = torch.arange(1, self.num_layers + 1, device=device).float()
        expected_layers = (halt_dist * layer_indices).sum(dim=1)
        
        probs = F.softmax(weighted_logits, dim=-1)
        softmax_conf = probs.max(dim=-1)[0].mean(dim=-1)
        comp_uncertainty = expected_layers / self.num_layers
        confidence = 0.7 * softmax_conf + 0.3 * (1 - comp_uncertainty)
        
        outputs = {
            'logits': weighted_logits,
            'probs': probs,
            'confidence': confidence,
            'expected_layers': expected_layers,
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
        
        # KL from geometric prior
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
        
        return {
            'total': task_loss + self.config.lambda_p * kl_loss,
            'task': task_loss,
            'kl': kl_loss
        }
    
    def generate_with_confidence(self, input_ids: torch.Tensor,
                                  max_new_tokens: int = 50,
                                  temperature: float = 1.0) -> Dict:
        """Generate text and track confidence/computation per token"""
        self.eval()
        generated = input_ids.clone()
        
        token_stats = []
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = self.forward(generated)
                
                next_logits = outputs['logits'][:, -1, :] / temperature
                next_probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(next_probs, num_samples=1)
                
                generated = torch.cat([generated, next_token], dim=1)
                
                token_stats.append({
                    'confidence': outputs['confidence'].item(),
                    'expected_layers': outputs['expected_layers'].item(),
                    'halt_dist': outputs['halt_dist'][0].cpu().numpy()
                })
                
                # Stop at EOS or very high confidence
                if next_token.item() == self.gpt2.config.eos_token_id:
                    break
        
        return {
            'generated_ids': generated,
            'token_stats': token_stats,
            'avg_confidence': np.mean([s['confidence'] for s in token_stats]),
            'avg_layers': np.mean([s['expected_layers'] for s in token_stats])
        }


# =============================================================================
# Evaluation: ECE for Language Models
# =============================================================================

def compute_lm_ece(model, dataloader, tokenizer, n_bins: int = 15) -> Dict[str, float]:
    """
    Compute ECE for language model predictions.
    
    For each token position, check if the top prediction was correct
    and compare to the model's confidence.
    """
    model.eval()
    
    all_confidences = []
    all_correct = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids']
            
            outputs = model(input_ids, labels=input_ids)
            
            # Get per-token predictions
            probs = outputs['probs'][:, :-1, :]  # Shift for next-token prediction
            targets = input_ids[:, 1:]
            
            # Token-level confidence and correctness
            token_conf = probs.max(dim=-1)[0]  # [B, T-1]
            token_pred = probs.argmax(dim=-1)  # [B, T-1]
            token_correct = (token_pred == targets).float()
            
            all_confidences.extend(token_conf.flatten().cpu().numpy())
            all_correct.extend(token_correct.flatten().cpu().numpy())
    
    confidences = np.array(all_confidences)
    correct = np.array(all_correct)
    
    # Compute ECE
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        if in_bin.sum() > 0:
            acc_bin = correct[in_bin].mean()
            conf_bin = confidences[in_bin].mean()
            ece += np.abs(acc_bin - conf_bin) * in_bin.mean()
    
    return {
        'ece': float(ece),
        'accuracy': float(correct.mean()),
        'avg_confidence': float(confidences.mean())
    }


# =============================================================================
# Demo: Test with GPT-2
# =============================================================================

def demo_ponder_gpt2():
    """Quick demo of PonderGPT2"""
    
    if not HAS_TRANSFORMERS:
        print("Cannot run demo: transformers not installed")
        return
    
    print("=" * 70)
    print("PONDER-GPT2 DEMO")
    print("=" * 70)
    
    # Load tokenizer and create model
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    config = PonderWrapperConfig(
        lambda_p=0.05,
        prior_p=0.3,
        freeze_base=True
    )
    
    model = PonderGPT2("gpt2", config)
    
    # Test forward pass
    print("\n--- Forward Pass Test ---")
    text = "The quick brown fox jumps over the"
    inputs = tokenizer(text, return_tensors="pt")
    
    outputs = model(inputs['input_ids'])
    
    print(f"Input: '{text}'")
    print(f"Confidence: {outputs['confidence'].item():.3f}")
    print(f"Expected layers: {outputs['expected_layers'].item():.2f} / {model.num_layers}")
    print(f"Halt distribution: {outputs['halt_dist'][0].detach().cpu().numpy().round(3)}")
    
    # Test generation
    print("\n--- Generation Test ---")
    gen_result = model.generate_with_confidence(
        inputs['input_ids'],
        max_new_tokens=10,
        temperature=0.8
    )
    
    generated_text = tokenizer.decode(gen_result['generated_ids'][0])
    print(f"Generated: '{generated_text}'")
    print(f"Avg confidence: {gen_result['avg_confidence']:.3f}")
    print(f"Avg layers: {gen_result['avg_layers']:.2f}")
    
    # Show per-token stats
    print("\n--- Per-Token Analysis ---")
    print(f"{'Token':<15} {'Conf':>8} {'Layers':>8}")
    print("-" * 35)
    
    generated_tokens = gen_result['generated_ids'][0, len(inputs['input_ids'][0]):]
    for i, (token_id, stats) in enumerate(zip(generated_tokens, gen_result['token_stats'])):
        token = tokenizer.decode([token_id])
        print(f"{repr(token):<15} {stats['confidence']:>8.3f} {stats['expected_layers']:>8.2f}")
    
    return model, tokenizer


if __name__ == "__main__":
    demo_ponder_gpt2()
