# Recommendation System with Exposure Bias Mitigation

This project implements a **Matrix Factorization-based Recommendation System** using **Bayesian Personalized Ranking (BPR)** loss, with a focus on studying and mitigating **exposure bias** in recommendation systems. The research investigates how popularity-based reweighting can improve robustness against dynamic exposure noise.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Experimental Setup](#experimental-setup)
- [Results](#results)
- [Understanding the Code](#understanding-the-code)
- [Dependencies](#dependencies)
- [Contributing](#contributing)

## 🎯 Project Overview

### What is Exposure Bias?
Exposure bias occurs when recommendation systems are trained on data that doesn't represent true user preferences, but rather what users were exposed to. Popular items get more exposure, creating a feedback loop that reinforces popularity bias.

### Research Question
This project investigates:
1. How does **dynamic exposure bias** (simulated noise during training) affect recommendation quality?
2. Can **popularity-based reweighting** mitigate the negative effects of exposure bias?

### Key Features
- **Matrix Factorization** with BPR loss for collaborative filtering
- **Dynamic exposure noise simulation** to study bias effects
- **Popularity-based reweighting** as a mitigation strategy
- **Comprehensive evaluation** with Recall@20 and NDCG@20 metrics
- **Robustness analysis** comparing clean vs. noisy training scenarios

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
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore rules
├── data/
│   └── ratings.csv          # Synthetic dataset (generated)
├── runs/                    # Experimental results
│   ├── static_base/         # Clean baseline results
│   ├── static_sol/          # Clean with reweighting results
│   ├── dyn_base/           # Noisy baseline results
│   ├── dyn_sol/            # Noisy with reweighting results
│   ├── summary.csv         # Aggregated results
│   └── robustness.csv      # Robustness analysis
├── train.py                # Main training script
├── analyze_results.py      # Results analysis script
└── make_data.py           # Dataset generation script
```

## 🔧 Usage

### Quick Start - Run All Experiments
```bash
# Generate data (if not already done)
python make_data.py

# Run all four experimental conditions
python train.py --model_dir runs/static_base --epochs 15
python train.py --model_dir runs/static_sol --reweight_type popularity --reweight_alpha 0.5 --epochs 15
python train.py --model_dir runs/dyn_base --noise_exposure_bias 0.3 --noise_schedule ramp --epochs 15
python train.py --model_dir runs/dyn_sol --noise_exposure_bias 0.3 --noise_schedule ramp --reweight_type popularity --reweight_alpha 0.5 --epochs 15

# Analyze results
python analyze_results.py
```

### Individual Experiment Commands

#### 1. Clean Baseline (No noise, no reweighting)
```bash
python train.py --model_dir runs/static_base --epochs 15
```

#### 2. Clean + Solution (No noise, with reweighting)
```bash
python train.py --model_dir runs/static_sol --reweight_type popularity --reweight_alpha 0.5 --epochs 15
```

#### 3. Noisy Baseline (With exposure noise, no reweighting)
```bash
python train.py --model_dir runs/dyn_base --noise_exposure_bias 0.3 --noise_schedule ramp --epochs 15
```

#### 4. Noisy + Solution (With exposure noise and reweighting)
```bash
python train.py --model_dir runs/dyn_sol --noise_exposure_bias 0.3 --noise_schedule ramp --reweight_type popularity --reweight_alpha 0.5 --epochs 15
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
| `--noise_exposure_bias` | `0.0` | Exposure bias noise level (0.0-1.0) |
| `--noise_schedule` | `none` | Noise schedule: `none` or `ramp` |
| `--reweight_type` | `none` | Reweighting strategy: `none` or `popularity` |
| `--reweight_alpha` | `0.0` | Popularity reweighting strength |

## 🧪 Experimental Setup

### Four Experimental Conditions

1. **Static Baseline**: Clean training data, no mitigation
2. **Static Solution**: Clean training data + popularity reweighting
3. **Dynamic Baseline**: Noisy training data (30% exposure bias), no mitigation
4. **Dynamic Solution**: Noisy training data + popularity reweighting

### Evaluation Metrics

- **Recall@20**: Fraction of relevant items in top-20 recommendations
- **NDCG@20**: Normalized Discounted Cumulative Gain at top-20
- **Robustness Drop**: `(clean_performance - noisy_performance) / clean_performance`

## 📊 Results

### Performance Summary

Based on the experimental results in `runs/summary.csv`:

| Condition | Recall@20 | NDCG@20 |
|-----------|-----------|---------|
| **Static Baseline** | 0.2024 | 0.0690 |
| **Static Solution** | 0.2014 | 0.0691 |
| **Dynamic Baseline** | 0.1734 | 0.0586 |
| **Dynamic Solution** | 0.1764 | 0.0586 |

### Key Findings

1. **Exposure Bias Impact**: Dynamic exposure noise reduces performance by ~14-15%
   - Recall@20 drops from 0.202 to 0.173 (14.3% decrease)
   - NDCG@20 drops from 0.069 to 0.059 (15.0% decrease)

2. **Mitigation Effectiveness**: Popularity reweighting provides modest improvements
   - In noisy conditions: Recall@20 improves from 0.173 to 0.176
   - Robustness drop reduces from 14.3% to 12.9% for Recall@20

3. **Robustness Analysis** (from `runs/robustness.csv`):
   - **Baseline Robustness Drop**: 14.3% (Recall), 15.0% (NDCG)
   - **Solution Robustness Drop**: 12.9% (Recall), 15.1% (NDCG)

### Interpreting Results

- **Lower robustness drop = better**: The solution reduces robustness drop for Recall@20
- **Practical impact**: While improvements are modest, they demonstrate the potential of reweighting strategies
- **Research implications**: Shows that simple popularity-based reweighting can partially mitigate exposure bias

## 🔍 Understanding the Code

### Core Components

#### 1. Matrix Factorization Model (`MF_BPR` class)
```python
class MF_BPR(nn.Module):
    def __init__(self, n_users, n_items, k=64):
        # Creates user and item embeddings
        self.U = nn.Embedding(n_users, k)    # User embeddings
        self.I = nn.Embedding(n_items, k)    # Item embeddings
```

#### 2. Exposure Bias Simulation
```python
def add_dynamic_exposure_noise(train_df, n_users, n_items, p, seed=42):
    # Adds p*|train| extra 'positive' clicks sampled by popularity
    # Simulates exposure bias where popular items get more fake interactions
```

#### 3. Popularity-Based Reweighting
```python
def build_pop_weights(train_df, n_items, alpha=0.5, eps=1e-6):
    # Creates weights inversely proportional to item popularity
    # Less popular items get higher weights during training
```

#### 4. BPR Loss Training
- **Bayesian Personalized Ranking**: Learns that observed items should rank higher than unobserved items
- **Triplet loss**: For user u, positive item i, negative item j: `loss = -log(σ(score(u,i) - score(u,j)))`

### Data Flow

1. **Data Generation** (`make_data.py`): Creates synthetic user-item interactions with popularity bias
2. **Training** (`train.py`): Trains MF model with optional noise and reweighting
3. **Evaluation**: Computes Recall@20 and NDCG@20 on test set
4. **Analysis** (`analyze_results.py`): Aggregates results and computes robustness metrics

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

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and commit: `git commit -am 'Add feature'`
4. Push to the branch: `git push origin feature-name`
5. Create a Pull Request

## 📝 License

This project is part of an IT thesis research. Please cite appropriately if using this code for academic purposes.

## 📧 Contact

For questions about this research or implementation, please open an issue in the GitHub repository.

---

**Note**: This is a research prototype designed for studying exposure bias in recommendation systems. For production use, consider additional optimizations and robustness measures.
