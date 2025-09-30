#!/usr/bin/env python3
"""
Prepare Gowalla and Amazon-book datasets for DCCF experiments.

This script downloads and preprocesses the datasets into the required format.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import requests
import gzip
import json
from urllib.parse import urlparse


def download_file(url, filepath):
    """Download a file from URL."""
    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(filepath, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    print(f"Downloaded to {filepath}")


def prepare_gowalla_dataset():
    """Prepare Gowalla dataset."""
    print("\n🌍 Preparing Gowalla Dataset...")
    
    # Create directory
    gowalla_dir = Path("data/gowalla")
    gowalla_dir.mkdir(parents=True, exist_ok=True)
    
    # Note: You'll need to manually download Gowalla dataset
    # from https://snap.stanford.edu/data/loc-gowalla.html
    print("📝 Gowalla Dataset Preparation Instructions:")
    print("1. Download 'loc-gowalla_totalCheckins.txt.gz' from:")
    print("   https://snap.stanford.edu/data/loc-gowalla.html")
    print("2. Extract to data/gowalla/loc-gowalla_totalCheckins.txt")
    print("3. Run this script again to process the data")
    
    raw_file = gowalla_dir / "loc-gowalla_totalCheckins.txt"
    if not raw_file.exists():
        print(f"❌ Raw file not found: {raw_file}")
        return False
    
    print("Processing Gowalla data...")
    
    # Read the raw data
    # Format: user_id, check-in_time, latitude, longitude, location_id
    df = pd.read_csv(raw_file, sep='\t', header=None, 
                     names=['userId', 'timestamp', 'latitude', 'longitude', 'itemId'])
    
    # Convert to implicit feedback (rating = 1 for all check-ins)
    df['rating'] = 1.0
    
    # Keep only required columns
    df = df[['userId', 'itemId', 'rating']]
    
    # Remove duplicates (same user-location pairs)
    df = df.drop_duplicates(subset=['userId', 'itemId'])
    
    # Filter users and items with minimum interactions
    min_interactions = 5
    
    # Filter users
    user_counts = df['userId'].value_counts()
    valid_users = user_counts[user_counts >= min_interactions].index
    df = df[df['userId'].isin(valid_users)]
    
    # Filter items
    item_counts = df['itemId'].value_counts()
    valid_items = item_counts[item_counts >= min_interactions].index
    df = df[df['itemId'].isin(valid_items)]
    
    # Save processed data
    output_file = gowalla_dir / "ratings.csv"
    df.to_csv(output_file, index=False)
    
    print(f"✅ Gowalla dataset prepared:")
    print(f"   - Users: {df['userId'].nunique():,}")
    print(f"   - Items: {df['itemId'].nunique():,}")
    print(f"   - Interactions: {len(df):,}")
    print(f"   - Saved to: {output_file}")
    
    return True


def prepare_amazon_book_dataset():
    """Prepare Amazon-book dataset."""
    print("\n📚 Preparing Amazon-book Dataset...")
    
    # Create directory
    amazon_dir = Path("data/amazon-book")
    amazon_dir.mkdir(parents=True, exist_ok=True)
    
    # Note: You'll need to manually download Amazon dataset
    print("📝 Amazon-book Dataset Preparation Instructions:")
    print("1. Download 'ratings_Books.csv' from:")
    print("   https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews")
    print("   OR")
    print("   http://jmcauley.ucsd.edu/data/amazon/ (Books category)")
    print("2. Place in data/amazon-book/ratings_Books.csv")
    print("3. Run this script again to process the data")
    
    raw_file = amazon_dir / "ratings_Books.csv"
    if not raw_file.exists():
        print(f"❌ Raw file not found: {raw_file}")
        return False
    
    print("Processing Amazon-book data...")
    
    # Read the raw data
    # Expected format: userId, itemId, rating, timestamp
    df = pd.read_csv(raw_file)
    
    # Standardize column names
    if 'user_id' in df.columns:
        df = df.rename(columns={'user_id': 'userId'})
    if 'item_id' in df.columns:
        df = df.rename(columns={'item_id': 'itemId'})
    if 'User_id' in df.columns:
        df = df.rename(columns={'User_id': 'userId'})
    if 'Item_id' in df.columns:
        df = df.rename(columns={'Item_id': 'itemId'})
    
    # Keep only required columns
    df = df[['userId', 'itemId', 'rating']]
    
    # Filter for high ratings (4-5 stars) to create implicit feedback
    df = df[df['rating'] >= 4.0]
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['userId', 'itemId'])
    
    # Filter users and items with minimum interactions
    min_interactions = 5
    
    # Filter users
    user_counts = df['userId'].value_counts()
    valid_users = user_counts[user_counts >= min_interactions].index
    df = df[df['userId'].isin(valid_users)]
    
    # Filter items
    item_counts = df['itemId'].value_counts()
    valid_items = item_counts[item_counts >= min_interactions].index
    df = df[df['itemId'].isin(valid_items)]
    
    # Sample if dataset is too large (for computational efficiency)
    max_interactions = 2000000  # 2M interactions max
    if len(df) > max_interactions:
        print(f"Sampling {max_interactions:,} interactions from {len(df):,}")
        df = df.sample(n=max_interactions, random_state=42)
    
    # Save processed data
    output_file = amazon_dir / "ratings.csv"
    df.to_csv(output_file, index=False)
    
    print(f"✅ Amazon-book dataset prepared:")
    print(f"   - Users: {df['userId'].nunique():,}")
    print(f"   - Items: {df['itemId'].nunique():,}")
    print(f"   - Interactions: {len(df):,}")
    print(f"   - Saved to: {output_file}")
    
    return True


def create_dataset_configs():
    """Create configuration files for the new datasets."""
    print("\n⚙️ Creating dataset configuration files...")
    
    # These were already created above, just confirm they exist
    configs = [
        "configs/datasets/gowalla_config.yaml",
        "configs/datasets/amazon_book_config.yaml"
    ]
    
    for config in configs:
        if os.path.exists(config):
            print(f"✅ Config exists: {config}")
        else:
            print(f"❌ Config missing: {config}")


def main():
    """Main preparation function."""
    print("🚀 Dataset Preparation for DCCF Robustness Study")
    print("=" * 60)
    
    # Create base data directory
    Path("data").mkdir(exist_ok=True)
    
    # Prepare datasets
    gowalla_success = prepare_gowalla_dataset()
    amazon_success = prepare_amazon_book_dataset()
    
    # Create configs
    create_dataset_configs()
    
    print("\n" + "=" * 60)
    print("📋 DATASET PREPARATION SUMMARY:")
    print(f"   - Gowalla: {'✅ Ready' if gowalla_success else '❌ Needs manual download'}")
    print(f"   - Amazon-book: {'✅ Ready' if amazon_success else '❌ Needs manual download'}")
    
    if gowalla_success and amazon_success:
        print("\n🎉 All datasets ready for experiments!")
        print("You can now run experiments with different datasets by updating the data_path in configs.")
    else:
        print("\n📝 Follow the instructions above to download the required datasets.")
        print("Then run this script again to process them.")


if __name__ == "__main__":
    main()
