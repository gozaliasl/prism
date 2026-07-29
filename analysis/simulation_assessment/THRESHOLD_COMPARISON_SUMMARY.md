# Threshold Comparison Summary

## Performance Metrics by Threshold

### EmpiricalLatest Batch

| Threshold | Accuracy | Precision | Recall | F1 Score | Best Metric |
|-----------|----------|-----------|--------|----------|-------------|
| 0.10      | 0.968    | 0.740     | 0.917  | 0.819    | Highest Recall |
| **0.15**  | **0.985**| **0.986** | **0.820**| **0.896** | ✅ **BEST F1** |
| 0.20      | 0.974    | 1.000     | 0.675  | 0.806    | Best Precision |
| 0.25      | 0.961    | 1.000     | 0.514  | 0.679    | - |
| 0.30      | 0.952    | 1.000     | 0.399  | 0.570    | - |

**Recommendation for EmpiricalLatest: Use threshold 0.15**
- Highest F1 score: **0.896**
- Excellent balance: Precision=0.986, Recall=0.820, Accuracy=0.985
- Provides near-perfect precision while maintaining good recall

---

### AblationLatest Batch

| Threshold | Accuracy | Precision | Recall | F1 Score | Best Metric |
|-----------|----------|-----------|--------|----------|-------------|
| 0.10      | 0.751    | 0.311     | 0.797  | 0.448    | Highest Recall |
| 0.15      | 0.823    | 0.393     | 0.740  | 0.514    | - |
| 0.20      | 0.873    | 0.498     | 0.634  | 0.558    | - |
| 0.25      | 0.895    | 0.594     | 0.525  | 0.557    | - |
| **0.30**  | **0.911**| **0.726** | **0.477**| **0.576** | ✅ **BEST F1** |

**Recommendation for AblationLatest: Use threshold 0.30**
- Highest F1 score: **0.576**
- Good balance: Precision=0.726, Recall=0.477, Accuracy=0.911
- Optimal trade-off for median noise configuration

---

## Summary

**Best thresholds for final_combo analysis:**
- **EmpiricalLatest**: **0.15** (F1=0.896, Precision=0.986, Recall=0.820)
- **AblationLatest**: **0.30** (F1=0.576, Precision=0.726, Recall=0.477)

**Note:** These thresholds should be used when running the final analysis to generate the best-quality figures for the paper. Use `--decision-thresholds 0.15 0.30` to set different thresholds for each batch.

