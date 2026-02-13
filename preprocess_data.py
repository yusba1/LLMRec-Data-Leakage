#!/usr/bin/env python3
"""
Data Pre-processing Script
This script implements the data splitting strategy described in the paper:
- Ratio: 7:1:2 (Train:Valid:Test)
- Logic: Global timestamp splitting (or random, depending on configuration)

Usage:
    python preprocess_data.py --dataset ml-1m --raw-path dataset/ml-1m/ratings.dat
"""

import pandas as pd
import numpy as np
import argparse
import os
from pathlib import Path

def load_ml1m(raw_path):
    # ML-1M format: UserID::MovieID::Rating::Timestamp
    print(f"Loading ML-1M from {raw_path}...")
    names = ['uid', 'iid', 'label', 'timestamp']
    try:
        df = pd.read_csv(raw_path, sep='::', header=None, names=names, engine='python')
        return df
    except Exception as e:
        print(f"Error loading ML-1M: {e}")
        return None

def load_amazon(raw_path):
    # Standard CSV format for Amazon
    print(f"Loading Amazon from {raw_path}...")
    try:
        df = pd.read_csv(raw_path)
        # Ensure columns match
        if 'userId' in df.columns: df.rename(columns={'userId': 'uid'}, inplace=True)
        if 'itemId' in df.columns: df.rename(columns={'itemId': 'iid'}, inplace=True)
        if 'rating' in df.columns: df.rename(columns={'rating': 'label'}, inplace=True)
        return df
    except Exception as e:
        print(f"Error loading Amazon: {e}")
        return None

def split_data(df, dataset_name, output_dir):
    print("Splitting data (7:1:2)...")
    
    # Sort by timestamp if available for strictly temporal split
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp')
    else:
        # Random shuffle if no timestamp
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    n = len(df)
    train_size = int(n * 0.7)
    valid_size = int(n * 0.1)
    
    train_data = df.iloc[:train_size]
    valid_data = df.iloc[train_size:train_size+valid_size]
    test_data = df.iloc[train_size+valid_size:]
    
    print(f"Train: {len(train_data)}, Valid: {len(valid_data)}, Test: {len(test_data)}")
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    train_data.to_pickle(os.path.join(output_dir, 'train_ood2.pkl'))
    valid_data.to_pickle(os.path.join(output_dir, 'valid_ood2.pkl'))
    test_data.to_pickle(os.path.join(output_dir, 'test_ood2.pkl'))
    
    print(f"Saved processed files to {output_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['ml-1m', 'amazon-book'])
    parser.add_argument('--raw-path', type=str, required=True, help='Path to raw dataset file')
    parser.add_argument('--output-dir', type=str, default=None)
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = os.path.join('dataset', args.dataset)
        
    if args.dataset == 'ml-1m':
        df = load_ml1m(args.raw_path)
    else:
        df = load_amazon(args.raw_path)
        
    if df is not None:
        split_data(df, args.dataset, args.output_dir)

if __name__ == '__main__':
    main()
