# Threshold Analysis Results (from temp_thr directories)

## Performance Metrics by Threshold

### EmpiricalLatest Batch

| Threshold | Accuracy | Precision | Recall | F1 Score | Recommendation |
|-----------|----------|-----------|--------|----------|----------------|
| 0.10      | 0.973    | 0.783     | 0.915  | 0.844    | High recall, moderate precision |
| **0.15**  | **0.986**| **0.986** | **0.834**| **0.904** | ✅ **BEST F1** - Optimal balance |
| 0.20      | 0.972    | 1.000     | 0.647  | 0.786    | Perfect precision, lower recall |
| 0.25      | 0.962    | 1.000     | 0.518  | 0.683    | Perfect precision, low recall |
| 0.30      | 0.952    | 1.000     | 0.399  | 0.570    | Perfect precision, very low recall |

**Best Threshold for EmpiricalLatest: 0.15**
- **F1 Score: 0.904** (highest)
- Precision: 0.986 (near-perfect)
- Recall: 0.834 (good)
- Accuracy: 0.986 (excellent)

---

### AblationLatest Batch

| Threshold | Accuracy | Precision | Recall | F1 Score | Recommendation |
|-----------|----------|-----------|--------|----------|----------------|
| 0.10      | 0.751    | 0.313     | 0.813  | 0.452    | High recall, low precision |
| 0.15      | 0.827    | 0.397     | 0.719  | 0.512    | Moderate balance |
| 0.20      | 0.865    | 0.473     | 0.613  | 0.534    | Moderate balance |
| 0.25      | 0.900    | 0.620     | 0.541  | 0.578    | Better precision-recall balance |
| **0.30**  | **0.917**| **0.778** | **0.477**| **0.591** | ✅ **BEST F1** - Best overall |

**Best Threshold for AblationLatest: 0.30**
- **F1 Score: 0.591** (highest)
- Precision: 0.778 (good)
- Recall: 0.477 (moderate)
- Accuracy: 0.917 (good)

---

## Summary and Recommendations

### Best Thresholds:
- **EmpiricalLatest**: **0.15** (F1=0.904, Precision=0.986, Recall=0.834)
- **AblationLatest**: **0.30** (F1=0.591, Precision=0.778, Recall=0.477)

### Key Observations:

1. **EmpiricalLatest** performs much better overall, achieving near-perfect precision (0.986) at threshold 0.15 while maintaining good recall (0.834).

2. **AblationLatest** requires a higher threshold (0.30) to optimize F1, trading off recall for precision. This reflects the more challenging nature of the median noise configuration.

3. The optimal threshold differs between batches, confirming that different noise configurations benefit from different threshold calibrations.

### Command to Run Final Analysis:

```bash
./venv/bin/python scripts/analyze_simulations.py \
  --batch-dirs \
    "/Volumes/extHD/jwst-lens-similator-output/batchA_empirical_20251028_150823_date_20251028_150855" \
    "/Volumes/extHD/jwst-lens-similator-output/batchB_ablation_median_20251028_150823_date_20251028_152200" \
  --batch-names EmpiricalLatest AblationLatest \
  --decision-thresholds 0.15 0.30 \
  --output-dir analysis/simulation_assessment/final_combo
```

This will generate the final high-quality figures using the optimized thresholds for each batch.

