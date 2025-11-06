"""
Create test datasets with 10%, 30%, and 50% MNAR missingness
MNAR: Missing Not at Random - missingness depends on the value itself
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

def create_mnar_missingness(df, target_percentage):
    """
    Create MNAR (Missing Not at Random) missingness
    - For numerical: Extreme values (top/bottom percentiles) more likely to be missing
    - For categorical: Rare categories more likely to be missing
    """
    df_missing = df.copy()
    total_cells = len(df) * len(df.columns)
    target_missing_cells = int((target_percentage / 100) * total_cells)
    
    # Track missing cells created
    missing_count = 0
    
    # Calculate missing cells per column (distribute across columns)
    missing_per_column = int(target_missing_cells / len(df.columns))
    
    for col in df.columns:
        if missing_count >= target_missing_cells:
            break
        
        col_data = df[col]
        
        # Skip if column is all NaN
        if col_data.isna().all():
            continue
        
        # Determine how many cells to make missing for this column
        remaining_needed = target_missing_cells - missing_count
        cells_for_this_col = min(missing_per_column, remaining_needed, len(df))
        
        if cells_for_this_col <= 0:
            continue
        
        # Check if column is numerical or categorical
        is_numerical = pd.api.types.is_numeric_dtype(col_data)
        
        if is_numerical:
            # MNAR for numerical: extreme values more likely to be missing
            # Remove NaN values for calculation
            valid_data = col_data.dropna()
            
            if len(valid_data) > 0:
                # Calculate thresholds
                lower_threshold = valid_data.quantile(0.1)  # Bottom 10%
                upper_threshold = valid_data.quantile(0.9)  # Top 10%
                
                # Create probability array
                probabilities = np.ones(len(df)) * 0.3  # Base probability
                
                for idx in df.index:
                    val = col_data.loc[idx]
                    if pd.notna(val):
                        # Extreme values get higher probability
                        if val <= lower_threshold or val >= upper_threshold:
                            probabilities[idx] = 0.7  # Higher probability for extremes
                        else:
                            probabilities[idx] = 0.1  # Lower probability for middle values
                
                # Normalize probabilities
                probabilities = probabilities / probabilities.sum()
                
                # Select indices based on probabilities
                n_select = min(cells_for_this_col, len(df))
                selected_indices = np.random.choice(
                    df.index,
                    size=n_select,
                    replace=False,
                    p=probabilities
                )
                
                # Make selected cells missing
                for idx in selected_indices:
                    if pd.notna(df_missing.loc[idx, col]):
                        df_missing.loc[idx, col] = np.nan
                        missing_count += 1
        
        else:
            # MNAR for categorical: rare categories more likely to be missing
            value_counts = col_data.value_counts()
            
            # Calculate rarity (inverse frequency)
            total_non_null = len(col_data.dropna())
            if total_non_null > 0:
                rarity = {}
                for val, count in value_counts.items():
                    rarity[val] = 1.0 / (count / total_non_null)  # Inverse frequency
                
                # Normalize rarity
                max_rarity = max(rarity.values()) if rarity else 1.0
                for val in rarity:
                    rarity[val] = rarity[val] / max_rarity if max_rarity > 0 else 0.5
                
                # Create probability array
                probabilities = np.ones(len(df)) * 0.3  # Base probability
                
                for idx in df.index:
                    val = col_data.loc[idx]
                    if pd.notna(val) and val in rarity:
                        # Rare values get higher probability
                        probabilities[idx] = 0.3 + (rarity[val] * 0.5)  # 0.3 to 0.8
                
                # Normalize probabilities
                probabilities = probabilities / probabilities.sum()
                
                # Select indices based on probabilities
                n_select = min(cells_for_this_col, len(df))
                selected_indices = np.random.choice(
                    df.index,
                    size=n_select,
                    replace=False,
                    p=probabilities
                )
                
                # Make selected cells missing
                for idx in selected_indices:
                    if pd.notna(df_missing.loc[idx, col]):
                        df_missing.loc[idx, col] = np.nan
                        missing_count += 1
    
    return df_missing

def create_test_missingness_datasets():
    """
    Create test datasets with 10%, 30%, and 50% missingness for all datasets
    """
    print("=" * 70)
    print("CREATING TEST DATASETS WITH MISSINGNESS (MNAR)")
    print("=" * 70)
    
    datasets = ['buy', 'phone', 'restaurant', 'zomato']
    missingness_levels = [10, 30, 50]
    
    output_dir = "test_sets_missing"
    os.makedirs(output_dir, exist_ok=True)
    
    for dataset_name in datasets:
        print(f"\n{'='*70}")
        print(f"PROCESSING {dataset_name.upper()}")
        print(f"{'='*70}")
        
        # Load original test data
        test_file = f"test_sets/{dataset_name}_test_original.csv"
        
        if not os.path.exists(test_file):
            print(f"ERROR: {test_file} not found")
            continue
        
        df = pd.read_csv(test_file)
        print(f"Original test data: {len(df)} rows, {len(df.columns)} columns")
        total_cells = len(df) * len(df.columns)
        print(f"Total cells: {total_cells}")
        
        # Create datasets with different missingness levels
        for missingness in missingness_levels:
            print(f"\n  Creating {missingness}% missingness (MNAR)...")
            
            # Create MNAR missingness
            df_missing = create_mnar_missingness(df, missingness)
            
            # Calculate actual missingness
            actual_missing = df_missing.isna().sum().sum()
            actual_percentage = (actual_missing / total_cells) * 100
            
            # Save file
            output_file = f"{output_dir}/{dataset_name}_test_{missingness}percent_missing.csv"
            df_missing.to_csv(output_file, index=False)
            
            print(f"    [OK] Saved: {output_file}")
            print(f"    Actual missing: {actual_missing}/{total_cells} ({actual_percentage:.1f}%)")
            
            # Show missing by column
            missing_by_col = df_missing.isna().sum()
            missing_by_col = missing_by_col[missing_by_col > 0].sort_values(ascending=False)
            
            if len(missing_by_col) > 0:
                print(f"    Missing by column (top 5):")
                for col, count in missing_by_col.head(5).items():
                    pct = (count / len(df)) * 100
                    print(f"      {col}: {count} ({pct:.1f}%)")

def verify_missingness():
    """
    Verify the created missing datasets
    """
    print(f"\n{'='*70}")
    print("VERIFICATION SUMMARY")
    print(f"{'='*70}")
    
    datasets = ['buy', 'phone', 'restaurant', 'zomato']
    missingness_levels = [10, 30, 50]
    
    print(f"{'Dataset':<12} {'Level':<8} {'Rows':<8} {'Missing':<10} {'%':<10}")
    print("-" * 70)
    
    for dataset_name in datasets:
        for level in missingness_levels:
            file = f"test_sets_missing/{dataset_name}_test_{level}percent_missing.csv"
            if os.path.exists(file):
                df = pd.read_csv(file)
                total_cells = len(df) * len(df.columns)
                missing_cells = df.isna().sum().sum()
                missing_pct = (missing_cells / total_cells) * 100
                print(f"{dataset_name:<12} {level}%{'':<4} {len(df):<8} {missing_cells:<10} {missing_pct:<10.1f}%")
            else:
                print(f"{dataset_name:<12} {level}%{'':<4} {'NOT FOUND':<8}")

if __name__ == "__main__":
    # Create missing test datasets
    create_test_missingness_datasets()
    
    # Verify
    verify_missingness()
    
    print(f"\n{'='*70}")
    print("COMPLETED: Test datasets with missingness created!")
    print(f"{'='*70}")
    print("Files saved in: test_sets_missing/")
    print("Format: {dataset}_test_{level}percent_missing.csv")





