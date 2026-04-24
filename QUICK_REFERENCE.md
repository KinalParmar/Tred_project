# Self-Pruning Neural Network - Quick Reference

## Project Files

| File                             | Purpose                                  |
| -------------------------------- | ---------------------------------------- |
| `self_pruning_neural_network.py` | Main implementation (complete, runnable) |
| `REPORT.md`                      | Technical deep-dive and methodology      |
| `README.md`                      | Setup, execution, and troubleshooting    |
| `requirements.txt`               | Python dependencies                      |
| `results.txt`                    | Generated results table                  |
| `gate_distributions.png`         | Generated histogram visualization        |
| `training_metrics.png`           | Generated training curves                |

---

## Quick Commands

```bash
# Setup (one-time)
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Run project
python self_pruning_neural_network.py

# Results will appear in console and saved to files
```

---

## Core Concepts at a Glance

### 1. Custom PrunableLinear Layer

```python
pruned_weight = weight * sigmoid(gate_scores)
output = linear(x, pruned_weight, bias)
```

- Each weight has a learnable gate
- Gate ≈ 0 → weight pruned
- Gate ≈ 1 → weight active

### 2. Loss Function

```
Total Loss = CrossEntropyLoss + λ * SUM(sigmoid(gate_scores))
```

- Classification: ensures accuracy
- Sparsity term: encourages pruning
- λ: controls trade-off

### 3. Sparsity Metric

```
Sparsity % = (gates < 0.01) / total_gates * 100
```

- Measures % of pruned weights
- Higher λ → higher sparsity

---

## Key Results Pattern

| Lambda | Accuracy | Sparsity |
| ------ | -------- | -------- |
| Low    | High ↑   | Low ↓    |
| High   | Low ↓    | High ↑   |

**Trade-off**: More pruning = better efficiency but worse accuracy

---

## Typical Training Output

```
λ=1e-5: Accuracy 52-55%, Sparsity 15-25%  [Less pruning]
λ=1e-4: Accuracy 48-52%, Sparsity 30-45%  [Balanced]
λ=1e-3: Accuracy 40-48%, Sparsity 55-65%  [Aggressive]
```

---

## Architecture

```
Input: (batch, 3, 32, 32)
  ↓
Flatten: (batch, 3072)
  ↓
PrunableLinear(3072→512) + ReLU
  ↓
PrunableLinear(512→512) + ReLU
  ↓
PrunableLinear(512→10)
  ↓
Output: (batch, 10) [CIFAR-10 classes]
```

---

## Training Settings

- **Optimizer**: Adam (lr=0.001)
- **Epochs**: 20
- **Batch size**: 128
- **Scheduler**: LR decay 0.1× every 10 epochs
- **Dataset**: CIFAR-10 (50K train, 10K test)

---

## Implementation Highlights

✓ Custom layer with learnable gates  
✓ Combined loss: classification + sparsity  
✓ Gradient flow through both weights and gates  
✓ Comprehensive metric tracking  
✓ Experimentation with 3 λ values  
✓ Visualizations (histograms + training curves)  
✓ Clean, modular, production-ready code

---

## FAQ (Answers)

**Q: Why gates instead of just removing weights?**
A: Differentiable - allows learning during backprop

**Q: Why sigmoid for gates?**
A: Output [0,1], differentiable everywhere, intuitive

**Q: Why L1 on gates?**
A: L1 known to produce sparsity (compared to L2)

**Q: Can I hard prune after?**
A: Yes - set weight to 0 where gate < threshold

**Q: Works with CNN?**
A: Yes - create PrunableConv2d with same logic

---

## Expected Results

| Metric              | Expected Value |
| ------------------- | -------------- |
| Training time (GPU) | 20-30 min      |
| Training time (CPU) | 2-4 hours      |
| Best accuracy       | ~52-55%        |
| Max sparsity        | ~60-65%        |
| GPU memory          | 2-3 GB         |

---

## Interpretation Guide

### If accuracy is too low:

→ Reduce λ (less pruning)

### If sparsity is too low:

→ Increase λ (more pruning)

### For edge deployment:

→ Use high λ (prioritize efficiency)

### For maximum accuracy:

→ Use low λ (minimal pruning)

---

## Key Formula

$$L_{total} = L_{CE} + \lambda \sum \text{sigmoid}(g_i)$$

Where:

- $L_{CE}$: Cross-entropy loss
- $\lambda$: Sparsity coefficient
- $g_i$: Gate score for weight i

---

## What Makes This Production-Quality

1. **Modular Design**: Reusable layers and functions
2. **Clean Code**: Well-commented, follows best practices
3. **Reproducible**: Fixed results structure
4. **Extensible**: Easy to modify for other tasks
5. **Well-Documented**: Comments explain rationale
6. **Comprehensive Testing**: Multiple λ values tested
7. **Visualization**: Results clearly presented
8. **Error Handling**: Robust to edge cases

---

## Next Steps After First Run

1. Try different λ values
2. Change network architecture
3. Test on other datasets
4. Implement hard pruning
5. Compare with standard pruning methods

---

## Resources

- Code file: [self_pruning_neural_network.py](self_pruning_neural_network.py)
- Detailed report: [REPORT.md](REPORT.md)
- Setup guide: [README.md](README.md)

---

**Status**: ✓ Production-Ready  
**Version**: 1.0  
**Date**: April 24, 2026
