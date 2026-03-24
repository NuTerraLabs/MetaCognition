# V2: Self-Supervised Metacognition via Predictive Consistency

## The Insight

V1 failed because of **weak supervision** — training a network to predict "am I right?" using binary labels leads to collapse. The gradient landscape has a strong attractor toward constant output.

V2 eliminates binary supervision entirely. Instead, we derive uncertainty from **self-supervised signals** that don't require knowing the answer:

### Three Pillars

1. **Predictive Consistency** — If a model is confident, its predictions should be stable under input perturbation. Instability → uncertainty.

2. **Representation Geometry** — Uncertain predictions cluster near decision boundaries. We can detect this from the geometry of the learned representation without labels.

3. **Progressive Self-Distillation** — The model teaches itself: predictions from the current epoch become soft targets for the next epoch's uncertainty estimator.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    V2: SELF-SUPERVISED METACOGNITION              │
├──────────────────────────────────────────────────────────────────┤
│                                                                    │
│  Input x ──┬──────────────────────────────────────────┐           │
│            │                                           │           │
│            ▼                                           ▼           │
│  ┌─────────────────┐                    ┌─────────────────────┐   │
│  │  Shared Encoder  │                    │  Augmented Views     │   │
│  │  h = Enc(x)     │                    │  x' = Aug(x)         │   │
│  └────────┬────────┘                    │  x''= Aug(x)         │   │
│           │                             └──────────┬──────────┘   │
│           ├─────────────────────┐                  │              │
│           ▼                     ▼                  ▼              │
│  ┌────────────────┐   ┌────────────────┐  ┌───────────────────┐  │
│  │  Classifier    │   │ Uncertainty    │  │ Consistency       │  │
│  │  ŷ = C(h)     │   │ Head           │  │ Estimator         │  │
│  │               │   │ û = U(h)       │  │ δ = |C(h')-C(h'')|│  │
│  └───────┬────────┘   └───────┬────────┘  └───────┬───────────┘  │
│          │                    │                    │              │
│          │              ┌─────┴─────┐              │              │
│          │              │  Self-Sup │◄─────────────┘              │
│          │              │  Target:  │                             │
│          │              │  û ≈ δ    │  ← Consistency AS target   │
│          │              └─────┬─────┘                             │
│          │                    │                                   │
│          ▼                    ▼                                   │
│  ┌─────────────────────────────────────┐                         │
│  │  Metacognitive Controller           │                         │
│  │  ŷ_final = (1-û)·ŷ + û·uniform     │                         │
│  │  (uncertain → push toward uniform)  │                         │
│  └─────────────────────────────────────┘                         │
│                                                                    │
└──────────────────────────────────────────────────────────────────┘
```

## Key Innovations

### 1. Consistency-Based Self-Supervision
Instead of "were you right?" (requires labels), we ask "are you stable?" (requires only the model itself):
```
uncertainty_target = E[|f(Aug(x)) - f(Aug(x))|]
```
A model that gives different answers to slightly different versions of the same input is **uncertain about that input**. No labels needed.

### 2. Geometric Uncertainty from Decision Boundaries
We compute distance to the nearest decision boundary in representation space using the classifier's weight vectors. Points near boundaries are inherently uncertain.

### 3. Progressive Self-Distillation
Every N epochs, the model's current predictions become soft targets for the uncertainty head:
- Epoch 0-10: Uncertainty trained only on consistency signal
- Epoch 10-20: Uncertainty also trained to predict KL divergence from self-distilled targets
- This bootstraps increasingly rich uncertainty signals from the model's own learning trajectory

### 4. Entropy Matching
The uncertainty head is trained to match the *entropy* of the classifier's predictions (a rich scalar signal) rather than binary correct/incorrect. High-entropy predictions → high uncertainty.

## Why This Should Avoid Collapse

| V1 Failure Mode | V2 Solution |
|-----------------|-------------|
| Binary supervision (weak gradient) | Continuous consistency signal (rich gradient) |
| Collapse to constant u | Consistency varies per-sample by construction |
| Representation entanglement | Separate augmentation pathway decouples signals |
| Predicting u≈accuracy trivially | No accuracy signal involved at all |

## Training Losses

```
L_total = L_task + α·L_consistency + β·L_geometry + γ·L_entropy + δ·L_distill

L_task: Standard cross-entropy (we still need the model to learn)
L_consistency: MSE(û, consistency_score) — self-supervised
L_geometry: MSE(û, boundary_distance_score) — self-supervised
L_entropy: MSE(û, normalized_entropy) — self-supervised
L_distill: KL(current_preds || ema_preds) — self-distillation
```

## Files

- `model.py` — Architecture definition
- `train.py` — Training pipeline with all self-supervised losses
- `experiment.py` — Full comparison against V1 baselines
