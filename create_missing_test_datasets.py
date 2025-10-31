"""
Create Missing Test Datasets with Different Missingness Levels
This script creates separate test files with 10%, 30%, and 50% missingness
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

def create_missing_test_data(original_file, output_dir, missingness_percentages=[10, 30, 50]):
    """
    Create test datasets with different missingness levels
    """
    print(f"Processing: {original_file}")
    
    # Load original dataset
    df = pd.read_csv(original_file)
    print(f"  Original: {len(df)} rows, {len(df.columns)} columns")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get dataset name from file path
    dataset_name = Path(original_file).stem.replace('_train_original', '').replace('_original', '')
    
    for missingness in missingness_percentages:
        print(f"  Creating {missingness}% missingness...")
        
        # Create a copy of the original data
        df_missing = df.copy()
        
        # Calculate number of missing values per column
        total_cells = len(df) * len(df.columns)
        missing_cells_per_col = int((missingness / 100) * len(df))
        
        # Apply missingness to each column
        for col in df.columns:
            if len(df) > 0:
                # Randomly select rows to make missing
                missing_indices = np.random.choice(
                    df.index, 
                    size=min(missing_cells_per_col, len(df)), 
                    replace=False
                )
                df_missing.loc[missing_indices, col] = np.nan
        
        # Save the missing dataset
        output_file = f"{output_dir}/{dataset_name}_test_{missingness}percent_missing.csv"
        df_missing.to_csv(output_file, index=False)
        
        # Calculate actual missingness
        actual_missing = df_missing.isna().sum().sum()
        actual_percentage = (actual_missing / total_cells) * 100
        
        print(f"    Saved: {output_file}")
        print(f"    Actual missing: {actual_missing}/{total_cells} ({actual_percentage:.1f}%)")
        
        # Show missing columns summary
        missing_cols = df_missing.isna().sum()
        missing_cols = missing_cols[missing_cols > 0].sort_values(ascending=False)
        
        if len(missing_cols) > 0:
            print(f"    Top missing columns:")
            for col, count in missing_cols.head(5).items():
                pct = (count / len(df)) * 100
                print(f"      {col}: {count} ({pct:.1f}%)")

def create_missing_test_datasets():
    """
    Create missing test datasets for all available datasets
    """
    print("CREATING MISSING TEST DATASETS")
    print("=" * 60)
    
    # Define datasets and their original files
    datasets = [
        {
            'name': 'phone',
            'original_file': 'train_sets/phone_train_original.csv',
            'output_dir': 'test_sets_missing'
        },
        {
            'name': 'buy',
            'original_file': 'train_sets/buy_train_original.csv',
            'output_dir': 'test_sets_missing'
        },
        {
            'name': 'restaurant',
            'original_file': 'train_sets/restaurant_train_original.csv',
            'output_dir': 'test_sets_missing'
        },
        {
            'name': 'zomato',
            'original_file': 'train_sets/zomato_train_original.csv',
            'output_dir': 'test_sets_missing'
        }
    ]
    
    # Process each dataset
    for dataset in datasets:
        if os.path.exists(dataset['original_file']):
            try:
                create_missing_test_data(
                    dataset['original_file'],
                    dataset['output_dir'],
                    missingness_percentages=[10, 30, 50]
                )
                print(f"SUCCESS: {dataset['name']} processed")
            except Exception as e:
                print(f"ERROR: {dataset['name']} failed - {e}")
        else:
            print(f"SKIP: {dataset['original_file']} not found")
        print()

def analyze_created_datasets():
    """
    Analyze the created missing datasets
    """
    print("ANALYZING CREATED MISSING DATASETS")
    print("=" * 60)
    
    test_dir = "test_sets_missing"
    if not os.path.exists(test_dir):
        print("No missing test datasets found!")
        return
    
    # Get all created files
    files = [f for f in os.listdir(test_dir) if f.endswith('.csv')]
    files.sort()
    
    print(f"Created {len(files)} missing test files:")
    print()
    
    for file in files:
        file_path = os.path.join(test_dir, file)
        try:
            df = pd.read_csv(file_path)
            
            # Calculate missing statistics
            total_cells = len(df) * len(df.columns)
            missing_cells = df.isna().sum().sum()
            missing_percentage = (missing_cells / total_cells) * 100
            
            # Count columns with missing values
            missing_cols = (df.isna().sum() > 0).sum()
            
            print(f"File: {file}")
            print(f"  Rows: {len(df)}, Columns: {len(df.columns)}")
            print(f"  Missing cells: {missing_cells}/{total_cells} ({missing_percentage:.1f}%)")
            print(f"  Columns with missing: {missing_cols}/{len(df.columns)}")
            
            # Show top missing columns
            missing_by_col = df.isna().sum().sort_values(ascending=False)
            missing_by_col = missing_by_col[missing_by_col > 0]
            
            if len(missing_by_col) > 0:
                print(f"  Top missing columns:")
                for col, count in missing_by_col.head(3).items():
                    pct = (count / len(df)) * 100
                    print(f"    {col}: {count} ({pct:.1f}%)")
            print()
            
        except Exception as e:
            print(f"ERROR reading {file}: {e}")

def main():
    """
    Main function
    """
    print("MISSING TEST DATASET CREATION")
    print("=" * 60)
    print("Creating test datasets with 10%, 30%, and 50% missingness")
    print("Using original full datasets as reference")
    print()
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create missing test datasets
    create_missing_test_datasets()
    
    # Analyze created datasets
    analyze_created_datasets()
    
    print("COMPLETED: Missing test datasets created successfully!")
    print("Files saved in: test_sets_missing/")

if __name__ == "__main__":
    main()







