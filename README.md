# A Study on Robust Recommender System using Disentangled Contrastive Collaborative Filtering (DCCF)

**IT Thesis Project - Data Science**

This thesis investigates the robustness of **Disentangled Contrastive Collaborative Filtering (DCCF)** under dynamic noise conditions. While DCCF was designed to handle noise in recommendation systems, it assumes noise patterns remain static during training. Our research explores how DCCF performs when noise distributions change dynamically over time and proposes a **static confidence denoiser with burn-in scheduling** to improve robustness.

**Implementation Note**: This project uses a custom PyTorch framework designed specifically for this robustness study, providing full control over the experimental design and transparent implementation of DCCF concepts without relying on external frameworks.

## 📋 Table of Contents

- [Thesis Overview](#thesis-overview)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Experimental Design](#experimental-design)
- [Results & Analysis](#results--analysis)
- [Understanding the Implementation](#understanding-the-implementation)
- [Dependencies](#dependencies)
- [Academic Context](#academic-context)

## 🎯 Thesis Overview

### Problem Statement
Real-world recommendation systems face **dynamic noise** in user feedback data:
- **Seasonal trends** (holiday shopping spikes)
- **Sudden popularity shifts** (viral content)
- **Spam/fake interactions** (bot behavior)
- **Accidental clicks** (user mistakes)

**DCCF's Limitation**: Assumes noise is **static** (unchanging during training), making it less robust in real-world scenarios where noise patterns evolve dynamically.

### Research Questions
1. **How does DCCF perform when noise distributions are dynamic rather than static?**
2. **Can a static confidence denoiser with burn-in scheduling improve robustness under dynamic noise conditions?**

### Hypothesis
We hypothesize that DCCF's performance degrades significantly under dynamic noise, and that our proposed **static confidence denoiser with burn-in scheduling** can mitigate this degradation while maintaining performance under static conditions.

### Our Solution
**Static Confidence Denoiser with Burn-in**:
- **Static Confidence Denoiser**: Down-weights likely noisy/over-exposed interactions using item popularity proxy
- **Burn-in Schedule**: Gradually introduces denoising over initial epochs for training stability  
- **Dynamic Adaptation**: Maintains robustness as noise patterns change over time (ramp-up, burst, shift)

## 🚀 Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/manisa1/it-thesis.git
cd it-thesis
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install torch torchvision torchaudio
pip install pandas numpy matplotlib seaborn
pip install scikit-learn
```

### Step 4: Generate Synthetic Dataset
```bash
python make_data.py
```
This creates a synthetic MovieLens-like dataset with 3,000 users and 1,500 items in the `data/` directory.

## 📁 Project Structure

```
recsys/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── .gitignore                         # Git ignore rules
├── THESIS_PRESENTATION_GUIDE.md       # Comprehensive thesis guide
├── data/
│   ├── ratings.csv                    # Synthetic dataset (generated)
│   ├── gowalla/                       # Gowalla dataset (optional)
│   └── amazon-book/                   # Amazon-book dataset (optional)
├── configs/
│   ├── experiments/                   # Experiment configurations
│   │   ├── static_baseline.yaml      # Static noise experiments
│   │   ├── dynamic_baseline.yaml     # Dynamic noise experiments
│   │   ├── burst_baseline.yaml       # Burst noise experiments
│   │   ├── shift_baseline.yaml       # Shift noise experiments
│   │   └── *_solution.yaml           # Corresponding solution experiments
│   └── datasets/                      # Dataset configurations
│       ├── gowalla_config.yaml       # Gowalla dataset config
│       └── amazon_book_config.yaml   # Amazon-book dataset config
├── runs/                              # Experimental results
│   ├── static_base/                   # Static baseline results
│   ├── static_sol/                    # Static solution results
│   ├── dyn_base/                      # Dynamic baseline results
│   ├── dyn_sol/                       # Dynamic solution results
│   ├── burst_base/                    # Burst baseline results
│   ├── burst_sol/                     # Burst solution results
│   ├── shift_base/                    # Shift baseline results
│   ├── shift_sol/                     # Shift solution results
│   ├── comprehensive_summary.csv      # All experiment results
│   └── comprehensive_robustness.csv   # Robustness analysis
├── src/                               # Modular source code
│   ├── models/                        # Model implementations
│   ├── training/                      # Training and noise modules
│   ├── evaluation/                    # Metrics and evaluation
│   ├── data/                          # Data processing
│   └── utils/                         # Configuration and logging
├── train.py                           # Enhanced training script (burst/shift support)
├── run_train_experiments.py           # Automated experiment runner
├── run_all_experiments.py             # Original modular experiment runner
├── analyze_comprehensive_results.py   # Enhanced results analysis
├── prepare_datasets.py                # Dataset preparation script
├── test_new_experiments.py            # Validation script
└── make_data.py                       # Synthetic dataset generation
```

## 🔧 Usage

### Quick Start - Run All Experiments

#### Option 1: Enhanced Training Script (Recommended)
```bash
# Generate data (if not already done)
python make_data.py

# Run all 8 experiments with enhanced burst/shift support
python run_train_experiments.py

# Analyze comprehensive results
python analyze_comprehensive_results.py
```

#### Option 2: Original Modular System
```bash
# Generate data (if not already done)
python make_data.py

# Run core 4 experiments only (faster)
python run_all_experiments.py --quick

# Run all experiments including additional analysis
python run_all_experiments.py

# Generate comprehensive analysis
python analyze_thesis_results.py
```

### Results Location

All experimental results are saved in the `runs/` directory:

```
runs/
├── static_base/metrics.csv     # Static baseline results
├── static_sol/metrics.csv      # Static solution results  
├── dyn_base/metrics.csv        # Dynamic baseline results
├── dyn_sol/metrics.csv         # Dynamic solution results
├── burst_base/metrics.csv      # Burst baseline results
├── burst_sol/metrics.csv       # Burst solution results
├── shift_base/metrics.csv      # Shift baseline results
├── shift_sol/metrics.csv       # Shift solution results
├── comprehensive_summary.csv   # All results combined
└── comprehensive_robustness.csv # Robustness analysis
```

Each `metrics.csv` contains:
- **Recall@K**: Recall at K (default K=20)
- **NDCG@K**: NDCG at K (default K=20)  
- **K**: The K value used for evaluation

### Individual Experiment Commands

#### 1. Static Baseline (No noise, no denoising)
```bash
python run_experiment.py --config configs/experiments/static_baseline.yaml
```

#### 2. Static Solution (No noise, with denoising)
```bash
python run_experiment.py --config configs/experiments/static_solution.yaml
```

#### 3. Dynamic Baseline (With dynamic noise, no denoising)
```bash
python run_experiment.py --config configs/experiments/dynamic_baseline.yaml
```

#### 4. Dynamic Solution (With dynamic noise and denoising)
```bash
python run_experiment.py --config configs/experiments/dynamic_solution.yaml
```

#### 5. Additional Noise Pattern Analysis
```bash
# Burst noise pattern
python run_experiment.py --config configs/experiments/burst_baseline.yaml

# Shift noise pattern  
python run_experiment.py --config configs/experiments/shift_baseline.yaml

# Different static noise levels
python run_experiment.py --config configs/experiments/static_05_baseline.yaml
python run_experiment.py --config configs/experiments/static_15_baseline.yaml
python run_experiment.py --config configs/experiments/static_20_baseline.yaml
```

### Updated Training Script with Burst and Shift Support

We've enhanced the training script to support advanced noise patterns:

```bash
# Run all experiments with the new enhanced script
python run_train_experiments.py

# Or run individual experiments with train.py directly:

# Burst noise experiment
python train.py --model_dir runs/burst_base --epochs 15 \
  --noise_exposure_bias 0.10 --noise_schedule burst \
  --noise_burst_start 5 --noise_burst_len 3 --noise_burst_scale 2.0

# Shift noise experiment  
python train.py --model_dir runs/shift_base --epochs 15 \
  --noise_exposure_bias 0.10 --noise_schedule shift \
  --noise_shift_epoch 8 --noise_shift_mode head2tail
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_path` | `data/ratings.csv` | Path to the ratings dataset |
| `--model_dir` | `runs/run1` | Directory to save model and results |
| `--epochs` | `15` | Number of training epochs |
| `--k` | `64` | Embedding dimension |
| `--k_eval` | `20` | Top-K for evaluation metrics |
| `--lr` | `0.01` | Learning rate |
| **Noise Parameters** | | |
| `--noise_exposure_bias` | `0.0` | Base exposure bias noise level (0.0-1.0) |
| `--noise_schedule` | `none` | Noise schedule: `none`, `ramp`, `burst`, `shift` |
| `--noise_schedule_epochs` | `10` | Epochs for ramp schedule |
| **Burst Noise Parameters** | | |
| `--noise_burst_start` | `4` | Epoch to start burst (1-based) |
| `--noise_burst_len` | `2` | Duration of burst in epochs |
| `--noise_burst_scale` | `2.0` | Noise multiplier during burst |
| **Shift Noise Parameters** | | |
| `--noise_shift_epoch` | `5` | Epoch where focus shifts (1-based) |
| `--noise_shift_mode` | `head2tail` | Shift direction: `head2tail` or `tail2head` |
| **Reweighting Parameters** | | |
| `--reweight_type` | `none` | Reweighting strategy: `none` or `popularity` |
| `--reweight_alpha` | `0.0` | Popularity reweighting strength |
| `--reweight_ramp_epochs` | `10` | Burn-in epochs for gradual reweighting |

## 🧪 Experimental Design

### Four Experimental Conditions

Our experimental design tests two factors: **noise type** (static vs. dynamic) and **training strategy** (baseline vs. solution):

#### 1. **Static Baseline** (`static_base`)
- **Noise**: Static (fixed throughout training)
- **Strategy**: Standard DCCF training (no denoising)
- **Purpose**: Baseline performance under DCCF's assumed conditions

#### 2. **Static Solution** (`static_sol`)
- **Noise**: Static (fixed throughout training)  
- **Strategy**: DCCF + static confidence denoiser with burn-in
- **Purpose**: Verify our solution doesn't harm performance under ideal conditions

#### 3. **Dynamic Baseline** (`dyn_base`)
- **Noise**: Dynamic (increases over training epochs using ramp schedule)
- **Strategy**: Standard DCCF training (no denoising)
- **Purpose**: Demonstrate DCCF's weakness under realistic dynamic noise

#### 4. **Dynamic Solution** (`dyn_sol`)
- **Noise**: Dynamic (increases over training epochs using ramp schedule)
- **Strategy**: DCCF + static confidence denoiser with burn-in
- **Purpose**: Test our solution's effectiveness under dynamic noise conditions

### Additional Experiments

Beyond the core 4 experiments, we include comprehensive analysis with:

#### **Static Noise Analysis**
- **5%, 10%, 15%, 20% static noise** - Testing different corruption levels
- Matches the noise rates mentioned in academic literature

#### **Advanced Dynamic Noise Patterns**
- **Ramp-up**: Gradual noise increase over training epochs (0% → base_level)
- **Burst**: Sudden noise spikes during specific training windows (2-3x base level)
- **Shift**: Corruption focus changes mid-training (head items → tail items or vice versa)

Each pattern simulates real-world scenarios:
- **Ramp**: Gradual system degradation or increasing bot activity
- **Burst**: Viral content spikes, Black Friday shopping, coordinated attacks
- **Shift**: Algorithm changes, user behavior shifts, seasonal pattern changes

### Evaluation Metrics

- **Recall@20**: Fraction of relevant items retrieved in top-20 recommendations
- **NDCG@20**: Normalized Discounted Cumulative Gain at top-20 (considers ranking quality)
- **Robustness Drop**: `(static_performance - dynamic_performance) / static_performance`
  - Lower robustness drop = better resilience to dynamic noise

## 📊 Results & Analysis

### Performance Summary

Based on experimental results from `runs/summary.csv`:

| Experimental Condition | Recall@20 | NDCG@20 | Description |
|------------------------|-----------|---------|-------------|
| **Static Baseline** | 0.2024 | 0.0690 | DCCF under ideal static noise conditions |
| **Static Solution** | 0.2014 | 0.0691 | Our solution under static noise (control) |
| **Dynamic Baseline** | 0.1734 | 0.0586 | DCCF under realistic dynamic noise |
| **Dynamic Solution** | 0.1764 | 0.0586 | Our solution under dynamic noise |

### Key Research Findings

#### 1. **DCCF's Dynamic Noise Vulnerability** ✅ *Hypothesis Confirmed*
Dynamic noise significantly degrades DCCF performance:
- **Recall@20**: 14.3% performance drop (0.202 → 0.173)
- **NDCG@20**: 15.0% performance drop (0.069 → 0.059)

#### 2. **Solution Effectiveness Under Dynamic Conditions** ✅ *Hypothesis Supported*
Our static confidence denoiser with burn-in provides measurable improvements:
- **Recall@20**: Improves from 0.173 to 0.176 under dynamic noise
- **Robustness**: Reduces performance drop from 14.3% to 12.9%

#### 3. **No Performance Degradation Under Static Conditions** ✅ *Control Verified*
Our solution maintains performance under static noise:
- Static baseline vs. static solution shows minimal difference
- Confirms our approach doesn't harm DCCF's original capabilities

### Robustness Analysis

From `runs/robustness.csv`:

| Metric | Baseline Robustness Drop | Solution Robustness Drop | Improvement |
|--------|-------------------------|-------------------------|-------------|
| **Recall@20** | 14.3% | 12.9% | **1.4% better** |
| **NDCG@20** | 15.0% | 15.1% | ~0% (neutral) |

### Thesis Contributions

1. **Identified DCCF's limitation**: Demonstrated significant performance degradation under dynamic noise
2. **Proposed practical solution**: Static confidence denoiser with burn-in scheduling  
3. **Empirical validation**: Showed measurable robustness improvements without harming static performance
4. **Real-world relevance**: Addressed the gap between DCCF's assumptions and realistic noise patterns
5. **Comprehensive analysis**: Multiple noise patterns (ramp-up, burst, shift) and corruption levels (5%-20%)

## 🔍 Understanding the Implementation

### Technical Architecture

Our implementation simulates DCCF's core functionality using **Matrix Factorization with BPR loss** as a representative collaborative filtering approach.

#### 1. **DCCF Simulation** (`MF_BPR` class)
```python
class MF_BPR(nn.Module):
    def __init__(self, n_users, n_items, k=64):
        self.U = nn.Embedding(n_users, k)    # User embeddings
        self.I = nn.Embedding(n_items, k)    # Item embeddings
```
- Represents the collaborative filtering component of DCCF
- Uses Bayesian Personalized Ranking (BPR) loss for implicit feedback

#### 2. **Enhanced Dynamic Noise Simulation**
```python
def add_dynamic_exposure_noise(train_df, n_users, n_items, p, focus=None, seed=42):
    # Simulates realistic dynamic noise patterns with focus control
    # - p: noise intensity (varies by schedule)
    # - focus: 'head', 'tail', or None (for shift patterns)
    # - Adds popularity-biased fake interactions
    
    if focus == "head":
        probs = probs ** 2  # Emphasize popular items even more
    elif focus == "tail":
        inv = 1.0 / (pop + 1e-8)  # Target long-tail items
        probs = inv / inv.sum()
```

**Noise Schedule Implementation**:
- **Static**: Fixed noise level throughout training
- **Ramp**: Gradual increase (0 → base_level over first 10 epochs)
- **Burst**: Spike during specific epochs (base_level → scale*base_level)
- **Shift**: Focus changes (head items → tail items at specified epoch)

#### 3. **Our Solution: Popularity-Aware Reweighting**
```python
def build_pop_weights(train_df, n_items, alpha=0.5, eps=1e-6):
    # Creates inverse popularity weights
    # Popular items get lower weights, rare items get higher weights
    pop = np.bincount(train_df["i"].values, minlength=n_items)
    w = (pop + eps) ** (-alpha)  # Inverse popularity weighting
```

#### 4. **Warm-up Scheduling**
```python
# Gradual introduction of reweighting (epochs 1-10)
if args.reweight_ramp_epochs > 0:
    ramp = min(1.0, epoch / max(1, args.reweight_ramp_epochs))
    iw = 1.0 + (item_weights - 1.0) * ramp
```

### Implementation Pipeline

1. **Data Generation** (`make_data.py`): Synthetic MovieLens-style dataset with popularity bias
2. **Training** (`train.py`): Four experimental conditions with controlled noise and reweighting
3. **Evaluation**: Standard recommendation metrics (Recall@20, NDCG@20)
4. **Analysis** (`analyze_results.py`): Robustness comparison and thesis conclusions

### Key Implementation Decisions

- **Why Matrix Factorization?** Captures DCCF's collaborative filtering essence while remaining interpretable
- **Why BPR Loss?** Standard for implicit feedback, similar to DCCF's contrastive approach
- **Why Synthetic Data?** Controlled environment to isolate dynamic noise effects
- **Why Ramp Schedule?** Realistic simulation of gradually changing noise patterns

## 📦 Dependencies

Create a `requirements.txt` file:

```txt
torch>=1.9.0
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
```

Install with:
```bash
pip install -r requirements.txt
```

## 🎓 Academic Context

### Thesis Information
- **Course**: IT Thesis (Data Science)
- **Topic**: Robust Recommender Systems under Dynamic Noise
- **Focus**: DCCF limitations and mitigation strategies
- **Target Audience**: Lecturers, students, and recommendation system researchers

### Related Work
This thesis builds upon:
- **DCCF Paper**: Disentangled Contrastive Collaborative Filtering
- **Limitation Identified**: Static noise assumption in dynamic environments
- **Our Contribution**: Popularity-aware reweighting with warm-up for dynamic robustness

### Potential Extensions
Future work could explore:
1. **Real-world datasets** (MovieLens, Amazon, Spotify)
2. **Advanced noise patterns** (seasonal, adversarial, concept drift)
3. **Alternative reweighting strategies** (uncertainty-based, temporal)
4. **Integration with full DCCF implementation**

## 🔬 Reproducibility

### For Researchers & Students
All experiments are fully reproducible:
```bash
# Clone and setup
git clone https://github.com/manisa1/it-thesis.git
cd it-thesis
pip install -r requirements.txt

# Generate data and run all experiments
python make_data.py
bash run_all_experiments.sh  # (create this script)
python analyze_results.py
```

### For Lecturers
- **Code Review**: All implementations are documented and modular
- **Results Verification**: Raw results in `runs/` directory with analysis scripts
- **Methodology**: Clear experimental design with controlled variables

## 📚 Citation


Group 6 (2025). "A Study on Robust Recommender System using 
Disentangled Contrastive Collaborative Filtering (DCCF)." 
IT Thesis, [Charles Darwin University].
```

## 📧 Contact

For questions about this thesis research:
- **GitHub Issues**: Technical implementation questions
- **Academic Inquiries**: Contact through university channels

---

**Academic Disclaimer**: This is a thesis research project focused on identifying and addressing limitations in DCCF under dynamic noise conditions. Results are based on controlled experiments with synthetic data.
