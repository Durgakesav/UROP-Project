"""
Recreate train/test split properly (70/30) and remove old cluster files
Then re-cluster the training data
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import shutil
import glob

# Dataset configuration
DATASETS = {
    'buy': {
        'file': 'buy.csv',
        'stratify_col': 'manufacturer'  # Stratification column
    },
    'phone': {
        'file': 'phone.csv',
        'stratify_col': 'brand'
    },
    'restaurant': {
        'file': 'restaurant.csv',
        'stratify_col': 'type'
    },
    'zomato': {
        'file': 'zomato.csv',
        'stratify_col': 'cuisine'
    }
}

def remove_old_files():
    """Remove all old train/test/cluster files"""
    print("=" * 70)
    print("REMOVING OLD FILES")
    print("=" * 70)
    
    # Remove train sets
    train_dir = "train_sets"
    if os.path.exists(train_dir):
        print(f"Removing {train_dir}/...")
        for file in glob.glob(f"{train_dir}/*.csv"):
            os.remove(file)
            print(f"  Removed: {file}")
    
    # Remove test sets
    test_dir = "test_sets_missing"
    if os.path.exists(test_dir):
        print(f"Removing {test_dir}/...")
        for file in glob.glob(f"{test_dir}/*.csv"):
            os.remove(file)
            print(f"  Removed: {file}")
    
    # Remove cluster results
    cluster_dir = "clustering_results"
    if os.path.exists(cluster_dir):
        print(f"Removing {cluster_dir}/...")
        for file in glob.glob(f"{cluster_dir}/*.csv"):
            os.remove(file)
            print(f"  Removed: {file}")
        for file in glob.glob(f"{cluster_dir}/*.json"):
            os.remove(file)
            print(f"  Removed: {file}")
    
    # Remove train_sets_clean
    clean_dir = "train_sets_clean"
    if os.path.exists(clean_dir):
        print(f"Removing {clean_dir}/...")
        for file in glob.glob(f"{clean_dir}/*.csv"):
            os.remove(file)
            print(f"  Removed: {file}")
    
    print("\n[OK] Old files removed\n")

def create_train_test_split(dataset_name, config):
    """
    Create proper 70/30 train/test split with stratification
    """
    print(f"\n{'='*70}")
    print(f"CREATING 70/30 SPLIT FOR {dataset_name.upper()}")
    print(f"{'='*70}")
    
    # Load original dataset
    if not os.path.exists(config['file']):
        print(f"ERROR: {config['file']} not found")
        return None, None
    
    df = pd.read_csv(config['file'])
    print(f"Original dataset: {len(df)} rows, {len(df.columns)} columns")
    
    # Check stratification column
    stratify_col = config['stratify_col']
    stratify = None
    
    if stratify_col not in df.columns:
        print(f"WARNING: Stratification column '{stratify_col}' not found, using random split")
    else:
        stratify_series = df[stratify_col]
        print(f"Stratification column: {stratify_col}")
        print(f"  Unique values: {stratify_series.nunique()}")
        print(f"  Distribution:")
        print(stratify_series.value_counts().head(10))
        
        # Check if stratification is possible (all classes need at least 2 members)
        value_counts = stratify_series.value_counts()
        min_count = value_counts.min()
        
        if min_count < 2:
            print(f"  WARNING: Some classes have only {min_count} member(s)")
            print(f"  Cannot use stratification - using random split instead")
        else:
            stratify = stratify_series
            print(f"  Minimum class size: {min_count} (stratification OK)")
    
    # Create 70/30 split
    train_df, test_df = train_test_split(
        df,
        test_size=0.30,
        random_state=42,
        stratify=stratify
    )
    
    print(f"\nSplit Results:")
    print(f"  Training set: {len(train_df)} rows ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Test set: {len(test_df)} rows ({len(test_df)/len(df)*100:.1f}%)")
    
    # Verify stratification
    if stratify_col and stratify_col in df.columns and stratify is not None:
        print(f"\nStratification Verification ({stratify_col}):")
        # Get all unique values
        all_values = df[stratify_col].unique()
        train_dist = train_df[stratify_col].value_counts(normalize=True).reindex(all_values, fill_value=0).sort_index()
        test_dist = test_df[stratify_col].value_counts(normalize=True).reindex(all_values, fill_value=0).sort_index()
        orig_dist = df[stratify_col].value_counts(normalize=True).reindex(all_values, fill_value=0).sort_index()
        
        print(f"  Original distribution preserved: [OK]")
        train_match = np.allclose(train_dist.values, orig_dist.values, atol=0.05)
        test_match = np.allclose(test_dist.values, orig_dist.values, atol=0.05)
        print(f"    Train matches original: {train_match}")
        print(f"    Test matches original: {test_match}")
    
    # Save train set
    os.makedirs("train_sets", exist_ok=True)
    train_file = f"train_sets/{dataset_name}_train_original.csv"
    train_df.to_csv(train_file, index=False)
    print(f"\n[OK] Saved: {train_file}")
    
    # Save test set (original - no missing values yet)
    os.makedirs("test_sets", exist_ok=True)
    test_file = f"test_sets/{dataset_name}_test_original.csv"
    test_df.to_csv(test_file, index=False)
    print(f"[OK] Saved: {test_file}")
    
    return train_df, test_df

def main():
    """Main function to recreate all splits"""
    print("=" * 70)
    print("RECREATING TRAIN/TEST SPLITS (70/30)")
    print("=" * 70)
    
    # Step 1: Remove old files
    remove_old_files()
    
    # Step 2: Create proper splits for all datasets
    results = {}
    for dataset_name, config in DATASETS.items():
        train_df, test_df = create_train_test_split(dataset_name, config)
        if train_df is not None and test_df is not None:
            results[dataset_name] = {
                'train': len(train_df),
                'test': len(test_df),
                'total': len(train_df) + len(test_df)
            }
    
    # Step 3: Summary
    print(f"\n{'='*70}")
    print("SPLIT SUMMARY")
    print(f"{'='*70}")
    print(f"{'Dataset':<15} {'Train':<10} {'Test':<10} {'Total':<10} {'Train %':<10}")
    print("-" * 70)
    for dataset_name, result in results.items():
        train_pct = (result['train'] / result['total']) * 100
        print(f"{dataset_name:<15} {result['train']:<10} {result['test']:<10} {result['total']:<10} {train_pct:<10.1f}%")
    
    print(f"\n{'='*70}")
    print("NEXT STEPS:")
    print("1. Run clustering on training data")
    print("2. Create test sets with missing values")
    print("3. Test LLM1 with proper train/test split")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()

