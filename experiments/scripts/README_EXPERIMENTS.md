# PoL-BFL Experiments Guide

This guide explains how to run all PoL-BFL experiments with statistical significance testing and generate publication-ready results.

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Experiment Overview](#experiment-overview)
- [Running Experiments](#running-experiments)
- [Statistical Analysis](#statistical-analysis)
- [Visualization](#visualization)
- [Paper Tables](#paper-tables)
- [Troubleshooting](#troubleshooting)

---

## 🚀 Quick Start

### Run All Experiments (Recommended)

```bash
cd PoL-BFL/Code/experiments/scripts

# Run all experiments with 3 seeds (default)
bash run_all_experiments.sh 3

# This will:
# 1. Run RQ1-RQ5 with 3 different random seeds
# 2. Aggregate results with statistical significance testing
# 3. Generate visualizations (PDF figures)
# 4. Generate LaTeX tables for paper
# 5. Create a summary report
```

### Run Individual Experiments

```bash
# RQ1: Security Evaluation
python runners/run_rq1_security.py --seed 42

# RQ2: Ablation Study
python runners/run_rq2_ablation.py --seed 42

# RQ3: Overhead Analysis
python runners/run_rq3_overhead.py --seed 42

# RQ4: Incentive Mechanism
python runners/run_rq4_incentive.py --seed 42

# RQ5: Composability (NEW!)
python runners/run_rq5_composability.py --seed 42
```

---

## 📊 Experiment Overview

### RQ1: Security Evaluation

**Research Question**: Can PoL-BFL effectively defend against Byzantine and free-riding attacks?

**Attacks Tested**:
- Byzantine: Random Noise, Label Flipping, Model Replacement, Gradient Inversion
- Free-riding: No Training, Lazy Training
- Sybil Attack

**Baselines**: Vanilla FL, Krum, Trimmed Mean, Median, PoL-BFL

**Metrics**: MA (Main Accuracy), TPR_conditional, TPR_e2e, FPR

### RQ2: Ablation Study

**Research Question**: What is the contribution of each component?

**Variants**:
- Vanilla FL (baseline)
- PoL Only
- PoL + ZKP
- PoL + Incentive
- Full System (PoL + ZKP + Incentive)

**Metrics**: MA, Detection Rate, Overhead

### RQ3: System Overhead

**Research Question**: What is the computational and communication overhead?

**Methods**: Vanilla FL, PoL-FL, PoL-FL+ZKP

**Metrics**: Training Time, Communication Cost, Storage Cost, ZKP Time, Gas Cost

### RQ4: Incentive Mechanism

**Research Question**: Does the incentive mechanism encourage honest participation?

**Scenarios**: No Incentive, Fixed Reward, Dynamic Reward, Sybil Attack

**Metrics**: Participation Rate, Attack Success Rate, Utility (Honest/Rational/Malicious)

### RQ5: Composability (NEW!)

**Research Question**: Can PoL seamlessly integrate with robust aggregation methods?

**Combinations**:
- Krum vs PoL + Krum
- Trimmed Mean vs PoL + Trimmed Mean
- Median vs PoL + Median
- Bulyan vs PoL + Bulyan

**Attacks**: Label Flipping, Data Poisoning (attacks that robust aggregation can detect but PoL cannot)

**Metrics**: MA, Improvement over standalone methods

---

## 🔬 Running Experiments

### Multi-Seed Experiments (Recommended)

For statistical significance, run each experiment with multiple seeds:

```bash
# Run RQ1 with 3 seeds
for seed in 42 123 456; do
    python runners/run_rq1_security.py \
        --seed $seed \
        --output_dir results/rq1_security
done

# Aggregate results
python aggregate_multi_seed_results.py \
    --experiment rq1 \
    --input_dir results/rq1_security \
    --pattern "*_seed_*.json"
```

### Custom Configuration

```bash
# RQ1 with custom dataset and rounds
python runners/run_rq1_security.py \
    --dataset MNIST \
    --num_rounds 20 \
    --seed 42

# RQ3 with custom overhead measurement
python runners/run_rq3_overhead.py \
    --dataset CIFAR10 \
    --num_rounds 100 \
    --seed 42
```

---

## 📈 Statistical Analysis

### Aggregate Multi-Seed Results

```bash
python aggregate_multi_seed_results.py \
    --experiment rq1 \
    --input_dir results/rq1_security \
    --output_dir results/rq1_security
```

**Output**:
- `rq1_aggregated.json`: Mean, std, confidence intervals, p-values
- `rq1_table.tex`: LaTeX table with significance markers

### Statistical Significance Testing

The aggregation script automatically:
1. Computes mean and standard deviation across seeds
2. Performs paired t-tests (PoL-BFL vs best baseline)
3. Adds significance markers:
   - `*`: p < 0.05
   - `**`: p < 0.01
   - `***`: p < 0.001

### Example Output

```json
{
  "byzantine_random_noise": {
    "PoL_FL": {
      "final_accuracy": {
        "mean": 0.923,
        "std": 0.005,
        "values": [0.921, 0.923, 0.925],
        "ci_95": [0.918, 0.928]
      },
      "significance_vs_best": {
        "baseline": "Trimmed_Mean",
        "t_statistic": 12.34,
        "p_value": 0.0001,
        "significant": true
      }
    }
  }
}
```

---

## 🎨 Visualization

### Generate All Figures

```bash
# RQ1: Accuracy curves + detection heatmap
python visualize_results.py \
    --experiment rq1 \
    --input results/rq1_security/rq1_aggregated.json \
    --output_dir results/rq1_security/figures

# RQ2: Ablation radar chart + bar chart
python visualize_results.py \
    --experiment rq2 \
    --input results/rq2_ablation/rq2_aggregated.json \
    --output_dir results/rq2_ablation/figures

# RQ3: Overhead comparison bars
python visualize_results.py \
    --experiment rq3 \
    --input results/rq3_overhead/rq3_aggregated.json \
    --output_dir results/rq3_overhead/figures

# RQ4: Utility evolution curves
python visualize_results.py \
    --experiment rq4 \
    --input results/rq4_incentive/rq4_aggregated.json \
    --output_dir results/rq4_incentive/figures

# RQ5: Composability comparison bars
python visualize_results.py \
    --experiment rq5 \
    --input results/rq5_composability/rq5_aggregated.json \
    --output_dir results/rq5_composability/figures
```

**Output**: Publication-quality PDF figures (300 DPI)

---

## 📝 Paper Tables

### Generate LaTeX Tables

```bash
# Generate all tables
python generate_paper_tables.py \
    --experiment all \
    --input results/rq1_security/rq1_aggregated.json \
    --output_dir results/tables

# Or generate individual tables
python generate_paper_tables.py \
    --experiment rq1 \
    --input results/rq1_security/rq1_aggregated.json \
    --output_dir results/tables
```

**Output**:
- `rq1_table.tex`: Security evaluation table
- `rq2_table.tex`: Ablation study table
- `rq3_table.tex`: Overhead analysis table
- `rq4_table.tex`: Incentive mechanism table
- `rq5_table.tex`: Composability table

### Table Features

- ✅ Mean ± Std formatting
- ✅ Statistical significance markers (*, **, ***)
- ✅ Bold formatting for best results
- ✅ Multi-row formatting for attacks
- ✅ Ready to copy-paste into LaTeX paper

---

## 🔧 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

```bash
# Reduce batch size
export BATCH_SIZE=32

# Or use CPU
export CUDA_VISIBLE_DEVICES=""
```

#### 2. Missing Dependencies

```bash
# Install required packages
pip install scipy matplotlib seaborn
```

#### 3. Gradient Inversion Attack Not Running

The Gradient Inversion attack is now enabled by default in RQ1. If you want to disable it:

```python
# In run_rq1_security.py, comment out:
# 'byzantine_gradient_inversion': {'malicious_ratios': [0.2]},
```

#### 4. Statistical Significance Not Computed

Make sure you run experiments with multiple seeds (at least 3):

```bash
bash run_all_experiments.sh 3  # Run with 3 seeds
```

---

## 📚 File Structure

```
experiments/
├── scripts/
│   ├── runners/
│   │   ├── run_rq1_security.py          # RQ1 experiment
│   │   ├── run_rq2_ablation.py          # RQ2 experiment
│   │   ├── run_rq3_overhead.py          # RQ3 experiment
│   │   ├── run_rq4_incentive.py         # RQ4 experiment
│   │   └── run_rq5_composability.py     # RQ5 experiment (NEW!)
│   ├── utils/
│   │   └── statistical_analysis.py      # Statistical utilities (NEW!)
│   ├── aggregate_multi_seed_results.py  # Aggregate results (NEW!)
│   ├── visualize_results.py             # Generate figures (NEW!)
│   ├── generate_paper_tables.py         # Generate LaTeX tables (NEW!)
│   ├── generate_summary_report.py       # Generate summary (NEW!)
│   ├── run_all_experiments.sh           # Run all experiments (NEW!)
│   └── README_EXPERIMENTS.md            # This file (NEW!)
└── results/
    ├── rq1_security/
    │   ├── rq1_results_seed_42.json
    │   ├── rq1_results_seed_123.json
    │   ├── rq1_results_seed_456.json
    │   ├── rq1_aggregated.json
    │   ├── figures/
    │   │   ├── rq1_accuracy_*.pdf
    │   │   └── rq1_detection_heatmap.pdf
    │   └── tables/
    │       └── rq1_table.tex
    └── ... (similar for RQ2-RQ5)
```

---

## 🎯 Recommended Workflow

### For Paper Submission

1. **Run all experiments with multiple seeds**:
   ```bash
   bash run_all_experiments.sh 3
   ```

2. **Review aggregated results**:
   ```bash
   cat results/summary_*.md
   ```

3. **Copy figures to paper**:
   ```bash
   cp results/rq*/figures/*.pdf ../Paper/figures/
   ```

4. **Copy tables to paper**:
   ```bash
   cp results/rq*/tables/*.tex ../Paper/tables/
   ```

5. **Update paper with results**:
   - Use aggregated JSON files for exact numbers
   - Include significance markers in tables
   - Reference figures in text

### For Debugging

1. **Run single experiment with one seed**:
   ```bash
   python runners/run_rq1_security.py --seed 42
   ```

2. **Check logs**:
   ```bash
   tail -f results/rq1_security/log_seed_42.txt
   ```

3. **Visualize results**:
   ```bash
   python visualize_results.py --experiment rq1 --input results/rq1_security/rq1_results_seed_42.json
   ```

---

## 📞 Support

For issues or questions:
1. Check this README
2. Review experiment logs in `results/*/log_*.txt`
3. Check code comments in runner scripts
4. Contact the development team

---

**Last Updated**: 2025-01-11

**Version**: 2.0 (Phase 1 & 2 Complete)

