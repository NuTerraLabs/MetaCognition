"""
Extended LLM PonderNet Training & Evaluation
============================================

Train halting heads more thoroughly and evaluate calibration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple
import sys
sys.path.insert(0, '/home/doom/MetaCognition/src')

from transformers import GPT2LMHeadModel, AutoTokenizer
from llm_ponder_wrapper import PonderGPT2, PonderWrapperConfig


def compute_ece(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        if in_bin.sum() > 0:
            acc_bin = correct[in_bin].mean()
            conf_bin = confidences[in_bin].mean()
            ece += np.abs(acc_bin - conf_bin) * in_bin.mean()
    return float(ece)


def get_training_data():
    """Wikipedia-style training sentences"""
    return [
        # Science
        "The speed of light in vacuum is approximately 299,792,458 meters per second.",
        "DNA carries genetic information in all living organisms.",
        "Photosynthesis converts carbon dioxide and water into glucose and oxygen.",
        "The periodic table organizes chemical elements by atomic number.",
        "Gravity is one of the four fundamental forces of nature.",
        "Neurons transmit electrical signals throughout the nervous system.",
        "The Earth rotates on its axis once every 24 hours.",
        "Evolution occurs through natural selection over many generations.",
        
        # History
        "The Roman Empire fell in 476 AD after centuries of decline.",
        "World War II ended in 1945 with the surrender of Germany and Japan.",
        "The Renaissance began in Italy during the 14th century.",
        "The Industrial Revolution transformed manufacturing and society.",
        "Ancient Egypt built the pyramids as tombs for pharaohs.",
        
        # Technology
        "Artificial intelligence simulates human cognitive processes in machines.",
        "The internet connects billions of devices worldwide through protocols.",
        "Machine learning algorithms improve performance through experience.",
        "Quantum computers use qubits instead of classical bits.",
        "Neural networks are inspired by biological brain structures.",
        
        # General
        "The weather today is sunny with a high temperature expected.",
        "Books provide knowledge and entertainment to readers everywhere.",
        "Music has been part of human culture for thousands of years.",
        "Mathematics is the language of science and engineering.",
        "Art expresses creativity and emotion through various mediums.",
    ]


def train_ponder_gpt2(model: PonderGPT2, texts: List[str], tokenizer, 
                      epochs: int = 20, lr: float = 5e-4) -> List[float]:
    """Train halting heads with proper PonderNet loss"""
    
    optimizer = torch.optim.AdamW(model.halting_heads.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_kl = 0
        
        # Shuffle texts each epoch
        np.random.shuffle(texts)
        
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
            
            optimizer.zero_grad()
            outputs = model(inputs['input_ids'], labels=inputs['input_ids'])
            
            loss = outputs['loss']['total']
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.halting_heads.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += outputs['loss']['task'].item()
            epoch_kl += outputs['loss']['kl'].item()
        
        scheduler.step()
        avg_loss = epoch_loss / len(texts)
        avg_kl = epoch_kl / len(texts)
        losses.append(avg_loss)
        
        if (epoch + 1) % 5 == 0:
            # Check expected layers
            with torch.no_grad():
                test_input = tokenizer("The quick brown fox", return_tensors="pt")
                out = model(test_input['input_ids'])
                exp_layers = out['expected_layers'].item()
            
            print(f"Epoch {epoch+1:3d}: task_loss={avg_loss:.4f}, kl={avg_kl:.4f}, exp_layers={exp_layers:.2f}")
    
    return losses


def evaluate_calibration(model: nn.Module, texts: List[str], tokenizer, 
                          use_ponder: bool = False) -> Dict[str, float]:
    """Evaluate token-level calibration"""
    model.eval()
    
    all_confs = []
    all_correct = []
    all_layers = []
    
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            input_ids = inputs['input_ids']
            
            if input_ids.shape[1] < 2:
                continue
            
            if use_ponder:
                outputs = model(input_ids)
                probs = outputs['probs']
                all_layers.append(outputs['expected_layers'].item())
            else:
                outputs = model(input_ids)
                probs = F.softmax(outputs.logits, dim=-1)
            
            pred_probs = probs[:, :-1, :]
            targets = input_ids[:, 1:]
            
            confidence = pred_probs.max(dim=-1)[0].squeeze()
            predictions = pred_probs.argmax(dim=-1).squeeze()
            correct = (predictions == targets.squeeze()).float()
            
            all_confs.extend(confidence.cpu().numpy().tolist())
            all_correct.extend(correct.cpu().numpy().tolist())
    
    confs = np.array(all_confs)
    correct = np.array(all_correct)
    
    results = {
        'ece': compute_ece(confs, correct),
        'accuracy': correct.mean(),
        'avg_confidence': confs.mean(),
        'conf_correct': confs[correct > 0.5].mean() if (correct > 0.5).sum() > 0 else 0,
        'conf_incorrect': confs[correct < 0.5].mean() if (correct < 0.5).sum() > 0 else 0,
    }
    
    if all_layers:
        results['avg_layers'] = np.mean(all_layers)
    
    return results


def main():
    print("=" * 70)
    print("EXTENDED PONDERGPT-2 TRAINING & EVALUATION")
    print("=" * 70)
    
    # Setup
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Models
    gpt2_standard = GPT2LMHeadModel.from_pretrained("gpt2")
    gpt2_standard.eval()
    
    ponder_config = PonderWrapperConfig(
        lambda_p=0.1,      # Higher KL weight to encourage early exit
        prior_p=0.4,       # Prior expecting ~2.5 layers
        freeze_base=True
    )
    gpt2_ponder = PonderGPT2("gpt2", ponder_config)
    
    # Training data
    train_texts = get_training_data() * 5  # 125 samples
    
    # Test data (different from training)
    test_texts = [
        "The moon orbits the Earth every 27 days approximately.",
        "Shakespeare wrote Hamlet in the early 1600s.",
        "Python is a popular programming language for data science.",
        "Climate change affects ecosystems around the world.",
        "The stock market closed higher today after positive reports.",
        "Coffee is one of the most traded commodities globally.",
        "The brain processes visual information in the occipital lobe.",
        "Democracy originated in ancient Greece over two thousand years ago.",
        "Electric vehicles are becoming more common on roads.",
        "Jazz music originated in New Orleans in the early 20th century.",
    ]
    
    # Evaluate BEFORE training
    print("\n--- BEFORE TRAINING ---")
    std_before = evaluate_calibration(gpt2_standard, test_texts, tokenizer, use_ponder=False)
    ponder_before = evaluate_calibration(gpt2_ponder, test_texts, tokenizer, use_ponder=True)
    
    print(f"Standard GPT-2 ECE: {std_before['ece']:.4f}")
    print(f"PonderGPT-2 ECE:    {ponder_before['ece']:.4f}, layers: {ponder_before['avg_layers']:.2f}")
    
    # Train
    print("\n--- TRAINING HALTING HEADS ---")
    train_ponder_gpt2(gpt2_ponder, train_texts, tokenizer, epochs=30, lr=1e-3)
    
    # Evaluate AFTER training
    print("\n--- AFTER TRAINING ---")
    ponder_after = evaluate_calibration(gpt2_ponder, test_texts, tokenizer, use_ponder=True)
    
    print(f"\nStandard GPT-2 ECE: {std_before['ece']:.4f}")
    print(f"PonderGPT-2 ECE:    {ponder_after['ece']:.4f}, layers: {ponder_after['avg_layers']:.2f}")
    
    # Summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"\n{'Metric':<25} {'Standard':>12} {'PonderNet':>12}")
    print("-" * 50)
    print(f"{'ECE':<25} {std_before['ece']:>12.4f} {ponder_after['ece']:>12.4f}")
    print(f"{'Accuracy':<25} {std_before['accuracy']:>12.4f} {ponder_after['accuracy']:>12.4f}")
    print(f"{'Avg Confidence':<25} {std_before['avg_confidence']:>12.4f} {ponder_after['avg_confidence']:>12.4f}")
    print(f"{'Conf (correct)':<25} {std_before['conf_correct']:>12.4f} {ponder_after['conf_correct']:>12.4f}")
    print(f"{'Conf (incorrect)':<25} {std_before['conf_incorrect']:>12.4f} {ponder_after['conf_incorrect']:>12.4f}")
    print(f"{'Avg Layers':<25} {'12.00':>12} {ponder_after['avg_layers']:>12.2f}")
    
    improvement = (std_before['ece'] - ponder_after['ece']) / std_before['ece'] * 100
    print(f"\nECE Improvement: {improvement:+.1f}%")
    
    if improvement > 0:
        print("✓ PonderNet IMPROVED calibration!")
    else:
        print("✗ PonderNet did not improve calibration")
    
    return std_before, ponder_after


if __name__ == "__main__":
    main()
