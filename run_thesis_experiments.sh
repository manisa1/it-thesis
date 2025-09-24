#!/bin/bash
# DCCF Robustness Thesis - Complete Experiment Runner
# This script runs all experiments required for the thesis

echo "🎓 DCCF Robustness Thesis - Experiment Runner"
echo "=============================================="
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 not found. Please install Python 3.7+"
    exit 1
fi

# Check if data exists, create if needed
if [ ! -f "data/ratings.csv" ]; then
    echo "📊 Generating synthetic dataset..."
    python3 make_data.py
    if [ $? -ne 0 ]; then
        echo "❌ Failed to generate data"
        exit 1
    fi
    echo "✅ Data generated successfully"
fi

# Install dependencies if needed
echo "📦 Checking dependencies..."
python3 -c "import torch, pandas, numpy, yaml" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing required packages..."
    pip3 install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install dependencies"
        exit 1
    fi
fi

echo "✅ Dependencies ready"
echo ""

# Run all experiments
echo "🚀 Starting thesis experiments..."
python3 run_all_experiments.py

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 THESIS EXPERIMENTS COMPLETED SUCCESSFULLY!"
    echo ""
    echo "📁 Results are available in:"
    echo "   - runs/summary.csv (main results table)"
    echo "   - runs/robustness.csv (robustness analysis)"
    echo "   - runs/*/metrics.csv (individual experiment results)"
    echo ""
    echo "📊 You can now use these results in your thesis!"
else
    echo ""
    echo "❌ Some experiments failed. Check the output above for details."
    exit 1
fi
