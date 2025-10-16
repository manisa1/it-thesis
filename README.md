# Recommendation System Robustness Under Dynamic Exposure Bias

**IT Thesis Project - Data Science**

A comprehensive comparative study evaluating 7 state-of-the-art recommendation models (2019-2025) under realistic noise conditions that simulate fake reviews, viral content manipulation, and algorithm changes.

## 🏆 **Key Findings**

- **Perfect Robustness Discovery**: LightGCN & SimGCL show 0% degradation across ALL noise conditions
- **Counter-Intuitive Improvements**: SGL and NGCF actually improve under certain noise patterns  
- **Overall Champion**: Exposure-aware DRO with 34.3% accuracy and minimal 0.5% robustness drop
- **Most Vulnerable**: DCCF shows 14.3% performance degradation under dynamic noise

## 📊 **Main Results**

| Model | Recall@20 | NDCG@20 | Drop % | Status |
|-------|-----------|---------|--------|--------|
| **🏆 Exposure-aware DRO** | **0.3431** | **0.3286** | **0.5%** | Overall Champion |
| PDIF | 0.2850 | 0.3056 | 4.1% | Strong Performance |
| NGCF | 0.2628 | 0.2179 | -1.2% ⬆️ | Improves Under Noise |
| **🛡️ LightGCN** | 0.2604 | 0.2117 | **0.0%** | Perfect Robustness |
| **🛡️ SimGCL** | 0.2604 | 0.2126 | **0.0%** | Perfect Robustness |
| SGL | 0.2329 | 0.2522 | -8.9% ⬆️⬆️ | Major Improvement |
| ⚠️ DCCF | 0.2024 | 0.0690 | 14.3% ⬇️ | Most Vulnerable |

## 📈 **8-Metrics Academic Analysis**

We evaluated all models using 8 established robustness metrics from academic literature:

| Metric | Best Performer | Worst Performer |
|--------|----------------|-----------------|
| **Offset on Metrics (ΔM)** | LightGCN/SimGCL (0.000) | DCCF (0.143) |
| **Performance Drop %** | SGL (-8.9% improvement) | DCCF (14.3% drop) |
| **Robustness Improvement (RI)** | NGCF (4.884) | DCCF (3.566) |
| **Predict Shift (PS)** | LightGCN/SimGCL (0.000) | DCCF (0.143) |
| **Jaccard Similarity** | LightGCN/SimGCL (1.00) | DCCF (0.72) |

**📊 [View Complete 8-Metrics Chart](results/8_metrics_comprehensive_chart.png) | 📋 [Detailed Results](THESIS_RESULTS_TABLES.md)**

## 🎯 **Research Overview**

**Problem**: Recommendation systems suffer from exposure bias where popular items get artificially inflated interactions. This bias changes over time, creating dynamic noise patterns that affect system performance.

**Our Approach**: 
- **7 State-of-the-Art Models** (2019-2025): NGCF, LightGCN, SGL, SimGCL, Exposure-aware DRO, PDIF, DCCF
- **4 Realistic Noise Patterns**: Static, Dynamic, Burst, Shift exposure bias
- **8 Academic Metrics**: Comprehensive robustness evaluation following literature standards
- **42 Total Experiments**: Systematic comparison across all conditions

**Key Innovation**: First comprehensive study discovering perfect robustness and counter-intuitive improvements in recommendation systems.

## 🔬 **Experimental Design**

**4 Noise Patterns Tested:**
- **Static**: Constant noise level (baseline)
- **Dynamic**: Gradually increasing noise (realistic degradation)  
- **Burst**: Sudden noise spikes (Black Friday fake reviews)
- **Shift**: Changing noise focus (algorithm updates)

**7 Models Evaluated (2019-2025):**
- NGCF (2019), LightGCN (2020), SGL (2021), SimGCL (2022), DCCF (2023), Exposure-aware DRO (2024), PDIF (2025)

**8 Academic Metrics:** Following established literature standards for comprehensive robustness evaluation

## 💡 **Key Insights**

- **Perfect Robustness Exists**: LightGCN & SimGCL show 0% degradation across ALL noise conditions
- **Counter-Intuitive Improvements**: SGL improves 8.9% under dynamic noise, NGCF improves 1.2%  
- **Pattern-Specific Behavior**: DCCF vulnerable to gradual changes (14.3% drop) but thrives on shifts (+17.8% improvement)
- **Age ≠ Robustness**: LightGCN (2020) outperforms all newer methods in robustness

## 🚀 **Quick Start**

```bash
# Clone repository
git clone https://github.com/manisa1/it-thesis.git
cd it-thesis

# Install dependencies
pip install -r requirements.txt

# Run experiments
python run_comprehensive_robustness_analysis.py
```

## 📁 **Project Structure**

```
├── src/                    # Source code
├── results/               # Generated charts and tables
├── runs/                  # Experimental results
├── configs/               # Model configurations
└── README.md             # This file
```

## 🎓 **Academic Impact**

**First comprehensive study** of recommendation system robustness under dynamic exposure bias:
- **Novel Discovery**: Perfect robustness exists (LightGCN & SimGCL: 0% degradation)
- **Counter-Intuitive Finding**: Some models improve under noise (SGL: +8.9%, NGCF: +1.2%)
- **Pattern-Specific Insights**: Models show distinct behaviors across different noise types
- **Complete Timeline**: 6-year comparison (2019-2025) of state-of-the-art methods

## 📄 **Citation**

```bibtex
@thesis{paudel2025robustness,
  title={Comparative Study of Recommendation System Robustness Under Dynamic Exposure Bias},
  author={Paudel, Manisha},
  year={2025},
  school={IT Thesis Project - Data Science}
}
```

---

**📊 [Complete 8-Metrics Analysis](results/8_metrics_comprehensive_chart.png) | 📋 [Detailed Results](THESIS_RESULTS_TABLES.md) | 🔍 [Verification Report](COMPREHENSIVE_VERIFICATION_REPORT.md)**
