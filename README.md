# Synthetic Metacognition: Can Neural Networks Learn Self-Assessment?

> **Status**: Active research. V1 results documented (mostly negative). V2 approach in development.

## What This Project Is

An honest investigation into whether neural networks can learn to monitor their own uncertainty — "synthetic metacognition." We tried 4 architectural approaches across V1. Most failed. We document why, and are now building a fundamentally different V2 approach.

## Key Results (V1)

| Approach | ECE Change | Verdict |
|----------|-----------|---------|
| Triadic (Monitor+Controller) | Worse | ❌ Monitor collapse |
| Neuro-symbolic (Evidential+RND+MoE) | 28.8% worse | ❌ Compounding failures |
| Temperature scaling (1 param) | 33–61% better | ✅ Simple baseline |
| Label smoothing | 36% better | ✅ Simple baseline |
| PonderNet | 70% better | ⚠️ Ensemble effect, not metacognition |

**Full details:** [PAPER.md](PAPER.md) (unified paper) · [FINDINGS_LOG.md](FINDINGS_LOG.md) (verified numbers)

## V2: New Approach

V2 attacks the core problem — **monitor collapse** — with self-supervised signals instead of binary correct/incorrect labels:

- **Consistency-based uncertainty:** Stability under augmentation as self-supervised target
- **Contrastive uncertainty:** Pull correct representations together, push uncertain ones apart — without labels
- **Progressive self-distillation:** Model teaches itself, using improving predictions as curriculum

```bash
cd /home/doom/MetaCognition/v2
pip install torch numpy scikit-learn
python train.py          # Run the full v2 pipeline
python experiment.py     # Compare v2 against all baselines
```

## Quick Start (V1)

```bash
cd /home/doom/MetaCognition
pip install torch numpy scikit-learn matplotlib seaborn tqdm pandas
python verify_all_claims.py         # Verify all claimed numbers
python experiments/reproducibility_test.py  # Multi-seed PonderNet
python experiments/test_neuro_symbolic.py   # Neuro-symbolic
```

## Project Structure

```
MetaCognition/
├── PAPER.md                     # ← Unified paper (all findings)
├── FINDINGS_LOG.md              # Verified results log
├── README.md                    # This file
├── verify_all_claims.py         # Reproducibility script
│
├── src/                         # V1 implementations
│   ├── models.py               # Triadic architecture (fails)
│   ├── ponder_net.py           # PonderNet (best ECE)
│   ├── neuro_symbolic.py       # Neuro-symbolic agent (fails)
│   ├── training.py             # Training utilities
│   ├── evaluation.py           # Metrics
│   └── ...
│
├── experiments/                 # V1 experiment scripts
│
├── v2/                          # ★ New approach
│   ├── README.md               # V2 design document
│   ├── model.py                # Self-supervised metacognitive architecture
│   ├── train.py                # Training pipeline
│   └── experiment.py           # Comparison experiments
│
└── archive/                     # Superseded papers and old files
```

## The Core Problem

**Monitor collapse**: Training a network to predict "am I right?" using binary labels leads to constant output — the gradient landscape has a strong attractor toward $u \approx \text{accuracy\_rate}$ for all inputs. V2 addresses this by eliminating binary supervision entirely.

## Citation

```bibtex
@misc{metacognition2026,
  title={Synthetic Metacognition: A Comprehensive Investigation into Neural Self-Assessment},
  author={Ismail Haddou},
  year={2026},
  note={Nu Terra Labs Ltd. See PAPER.md for full results.}
}
```
