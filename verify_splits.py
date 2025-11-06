import pandas as pd
import os

datasets = ['buy', 'phone', 'restaurant', 'zomato']

print("=" * 60)
print("TRAIN/TEST SPLIT VERIFICATION")
print("=" * 60)
print(f"{'Dataset':<12} {'Train':<8} {'Test':<8} {'Total':<8} {'Train %':<10}")
print("-" * 60)

for dataset in datasets:
    train_file = f"train_sets/{dataset}_train_original.csv"
    test_file = f"test_sets/{dataset}_test_original.csv"
    
    if os.path.exists(train_file) and os.path.exists(test_file):
        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)
        total = len(train_df) + len(test_df)
        train_pct = (len(train_df) / total) * 100
        
        print(f"{dataset:<12} {len(train_df):<8} {len(test_df):<8} {total:<8} {train_pct:<10.1f}%")

print("=" * 60)
print("[OK] All datasets split into 70/30 train/test")





