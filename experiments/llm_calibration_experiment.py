"""
LLM Calibration Experiment: Does PonderNet Improve LLM Calibration?
===================================================================

This experiment compares calibration between:
1. Standard GPT-2 (softmax confidence)
2. PonderGPT-2 (computation-based confidence)

We measure ECE on next-token prediction across different text types.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Install transformers: pip install transformers")

import sys
sys.path.insert(0, '/home/doom/MetaCognition/src')
from llm_ponder_wrapper import PonderGPT2, PonderWrapperConfig


def compute_token_ece(confidences: np.ndarray, correct: np.ndarray, 
                      n_bins: int = 15) -> float:
    """Compute ECE for token-level predictions"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        if in_bin.sum() > 0:
            acc_bin = correct[in_bin].mean()
            conf_bin = confidences[in_bin].mean()
            ece += np.abs(acc_bin - conf_bin) * in_bin.mean()
    
    return float(ece)


@dataclass
class CalibrationResult:
    """Results from calibration evaluation"""
    ece: float
    accuracy: float
    avg_confidence: float
    confidence_correct: float
    confidence_incorrect: float
    

def evaluate_gpt2_calibration(model: nn.Module, texts: List[str], 
                               tokenizer, use_ponder: bool = False) -> CalibrationResult:
    """Evaluate calibration of a GPT-2 model on given texts"""
    model.eval()
    
    all_confs = []
    all_correct = []
    
    with torch.no_grad():
        for text in texts:
            # Tokenize
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            input_ids = inputs['input_ids']
            
            if input_ids.shape[1] < 2:
                continue
            
            if use_ponder:
                outputs = model(input_ids)
                probs = outputs['probs']
            else:
                outputs = model(input_ids)
                probs = F.softmax(outputs.logits, dim=-1)
            
            # Get predictions and targets (next token prediction)
            pred_probs = probs[:, :-1, :]  # [1, T-1, V]
            targets = input_ids[:, 1:]      # [1, T-1]
            
            # Confidence = max probability
            confidence = pred_probs.max(dim=-1)[0].squeeze()  # [T-1]
            predictions = pred_probs.argmax(dim=-1).squeeze()  # [T-1]
            correct = (predictions == targets.squeeze()).float()  # [T-1]
            
            all_confs.extend(confidence.cpu().numpy().tolist())
            all_correct.extend(correct.cpu().numpy().tolist())
    
    confs = np.array(all_confs)
    correct = np.array(all_correct)
    
    ece = compute_token_ece(confs, correct)
    
    return CalibrationResult(
        ece=ece,
        accuracy=correct.mean(),
        avg_confidence=confs.mean(),
        confidence_correct=confs[correct > 0.5].mean() if (correct > 0.5).sum() > 0 else 0,
        confidence_incorrect=confs[correct < 0.5].mean() if (correct < 0.5).sum() > 0 else 0
    )


def create_test_texts() -> Dict[str, List[str]]:
    """Create test texts of varying difficulty"""
    return {
        "simple": [
            "The cat sat on the mat.",
            "She went to the store.",
            "He likes to play basketball.",
            "The sun rises in the east.",
            "Water is wet and fire is hot.",
            "I have a dog named Max.",
            "The book is on the table.",
            "She reads books every day.",
        ],
        "moderate": [
            "The implications of quantum mechanics extend far beyond physics.",
            "Machine learning models require careful hyperparameter tuning.",
            "The Renaissance marked a significant cultural transformation in Europe.",
            "Photosynthesis converts carbon dioxide into glucose using sunlight.",
            "The stock market fluctuates based on investor sentiment and economic data.",
            "Neuroplasticity allows the brain to reorganize throughout life.",
        ],
        "complex": [
            "The phenomenological reduction brackets our natural attitude toward the world.",
            "Gödel's incompleteness theorems demonstrate inherent limitations of formal systems.",
            "The epistemological status of scientific theories remains philosophically contested.",
            "Quantum entanglement violates Bell inequalities, ruling out local hidden variables.",
            "The hermeneutic circle describes the interdependence of parts and wholes in interpretation.",
        ],
        "factual": [
            "Paris is the capital of France.",
            "The Earth orbits around the Sun.",
            "Water freezes at zero degrees Celsius.",
            "Albert Einstein developed the theory of relativity.",
            "DNA contains the genetic instructions for life.",
        ],
        "creative": [
            "Once upon a time in a land far away, there lived a",
            "The mysterious door opened to reveal a",
            "She looked up at the stars and wondered about",
            "In the year 2150, humanity had finally",
            "The ancient prophecy spoke of a hero who would",
        ]
    }


def run_calibration_comparison():
    """Compare calibration between standard GPT-2 and PonderGPT-2"""
    
    if not HAS_TRANSFORMERS:
        print("Cannot run: transformers not installed")
        return
    
    print("=" * 70)
    print("LLM CALIBRATION COMPARISON: GPT-2 vs PonderGPT-2")
    print("=" * 70)
    
    # Load models
    print("\nLoading models...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Standard GPT-2
    gpt2_standard = GPT2LMHeadModel.from_pretrained("gpt2")
    gpt2_standard.eval()
    
    # PonderGPT-2 (untrained halters)
    ponder_config = PonderWrapperConfig(lambda_p=0.05, prior_p=0.3, freeze_base=True)
    gpt2_ponder = PonderGPT2("gpt2", ponder_config)
    gpt2_ponder.eval()
    
    # Quick training of halters on some text
    print("\nTraining halting heads...")
    train_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming how we interact with technology.",
        "Climate change poses significant challenges for future generations.",
        "The human brain contains approximately 86 billion neurons.",
        "Shakespeare wrote many famous plays including Hamlet and Macbeth.",
    ] * 10  # Repeat for more training data
    
    optimizer = torch.optim.Adam(gpt2_ponder.halting_heads.parameters(), lr=1e-3)
    
    for epoch in range(5):
        total_loss = 0
        for text in train_texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
            
            optimizer.zero_grad()
            outputs = gpt2_ponder(inputs['input_ids'], labels=inputs['input_ids'])
            loss = outputs['loss']['total']
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 2 == 0:
            print(f"  Epoch {epoch+1}: loss = {total_loss/len(train_texts):.4f}")
    
    # Get test texts
    test_texts = create_test_texts()
    
    # Evaluate both models
    print("\n" + "=" * 70)
    print("CALIBRATION RESULTS")
    print("=" * 70)
    
    all_results = {}
    
    for category, texts in test_texts.items():
        print(f"\n--- {category.upper()} TEXTS ---")
        
        # Standard GPT-2
        result_std = evaluate_gpt2_calibration(gpt2_standard, texts, tokenizer, use_ponder=False)
        
        # PonderGPT-2
        result_ponder = evaluate_gpt2_calibration(gpt2_ponder, texts, tokenizer, use_ponder=True)
        
        all_results[category] = {'standard': result_std, 'ponder': result_ponder}
        
        print(f"{'Metric':<25} {'Standard':>12} {'PonderNet':>12} {'Change':>12}")
        print("-" * 65)
        print(f"{'ECE':<25} {result_std.ece:>12.4f} {result_ponder.ece:>12.4f} {(result_std.ece - result_ponder.ece)/result_std.ece*100:>+11.1f}%")
        print(f"{'Accuracy':<25} {result_std.accuracy:>12.4f} {result_ponder.accuracy:>12.4f}")
        print(f"{'Avg Confidence':<25} {result_std.avg_confidence:>12.4f} {result_ponder.avg_confidence:>12.4f}")
        print(f"{'Conf (correct)':<25} {result_std.confidence_correct:>12.4f} {result_ponder.confidence_correct:>12.4f}")
        print(f"{'Conf (incorrect)':<25} {result_std.confidence_incorrect:>12.4f} {result_ponder.confidence_incorrect:>12.4f}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    avg_std_ece = np.mean([r['standard'].ece for r in all_results.values()])
    avg_ponder_ece = np.mean([r['ponder'].ece for r in all_results.values()])
    
    print(f"\nAverage ECE (Standard GPT-2):  {avg_std_ece:.4f}")
    print(f"Average ECE (PonderGPT-2):     {avg_ponder_ece:.4f}")
    print(f"ECE Improvement:               {(avg_std_ece - avg_ponder_ece)/avg_std_ece*100:+.1f}%")
    
    # Analyze computation patterns
    print("\n" + "=" * 70)
    print("COMPUTATION ANALYSIS (PonderGPT-2)")
    print("=" * 70)
    
    with torch.no_grad():
        for category, texts in test_texts.items():
            layers_used = []
            for text in texts:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
                outputs = gpt2_ponder(inputs['input_ids'])
                layers_used.append(outputs['expected_layers'].item())
            
            avg_layers = np.mean(layers_used)
            print(f"{category.capitalize():>12}: {avg_layers:.2f} layers (of 12)")
    
    return all_results


if __name__ == "__main__":
    run_calibration_comparison()
