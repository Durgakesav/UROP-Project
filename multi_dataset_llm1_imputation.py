"""
Multi-Dataset LLM1 Imputation with User Selection
This script tests Gemini LLM1 on different datasets, shows missing columns,
and allows user to select which column to impute
"""

import pandas as pd
import json
import os
from scripts.gemini_llm1_pipeline import GeminiLLM1Pipeline

# Your Gemini API key
GEMINI_API_KEY = "AIzaSyC6yMHoOO60kywdWgBWcwUtB2nbTo8_kAg"

def analyze_dataset_missing_values(dataset_name, test_file, train_file):
    """
    Analyze missing values in a dataset
    """
    print(f"\n{'='*80}")
    print(f"ANALYZING DATASET: {dataset_name.upper()}")
    print(f"{'='*80}")
    
    try:
        # Load datasets
        test_df = pd.read_csv(test_file)
        train_df = pd.read_csv(train_file)
        
        print(f"Dataset Info:")
        print(f"  Training: {len(train_df)} rows, {len(train_df.columns)} columns")
        print(f"  Test: {len(test_df)} rows, {len(test_df.columns)} columns")
        
        # Analyze missing values
        print(f"\nMissing Values Analysis:")
        print(f"{'Column':<25} {'Missing':<10} {'Percentage':<12} {'Status'}")
        print(f"{'-'*60}")
        
        missing_columns = []
        for col in test_df.columns:
            missing_count = test_df[col].isna().sum()
            missing_pct = (missing_count / len(test_df)) * 100
            status = "HIGH" if missing_pct > 50 else "MEDIUM" if missing_pct > 20 else "LOW"
            
            print(f"{col:<25} {missing_count:<10} {missing_pct:<11.1f}% {status}")
            
            if missing_count > 0:
                missing_columns.append({
                    'column': col,
                    'missing_count': missing_count,
                    'percentage': missing_pct,
                    'status': status
                })
        
        return missing_columns, test_df, train_df
        
    except Exception as e:
        print(f"ERROR: Could not analyze dataset {dataset_name}: {e}")
        return [], None, None

def get_user_column_selection(missing_columns):
    """
    Let user select which column to impute
    """
    if not missing_columns:
        print("No missing columns found!")
        return None
    
    print(f"\n{'='*60}")
    print("SELECT COLUMN FOR IMPUTATION")
    print(f"{'='*60}")
    print("Available columns with missing values:")
    print(f"{'#':<3} {'Column':<20} {'Missing':<10} {'Percentage':<12}")
    print(f"{'-'*50}")
    
    for i, col_info in enumerate(missing_columns, 1):
        print(f"{i:<3} {col_info['column']:<20} {col_info['missing_count']:<10} {col_info['percentage']:<11.1f}%")
    
    while True:
        try:
            choice = input(f"\nSelect column number (1-{len(missing_columns)}): ").strip()
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(missing_columns):
                    selected_column = missing_columns[idx]['column']
                    print(f"Selected: {selected_column}")
                    return selected_column
            print("Invalid choice. Please try again.")
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
            return None

def run_llm1_imputation(dataset_name, test_df, train_df, target_column):
    """
    Run Gemini LLM1 imputation on selected column
    """
    print(f"\n{'='*80}")
    print(f"RUNNING GEMINI LLM1 IMPUTATION")
    print(f"Dataset: {dataset_name.upper()}")
    print(f"Target Column: {target_column}")
    print(f"{'='*80}")
    
    try:
        # Initialize pipeline
        pipeline = GeminiLLM1Pipeline(GEMINI_API_KEY)
        
        # Find first missing row for the target column
        missing_rows = test_df[test_df[target_column].isna()]
        if len(missing_rows) == 0:
            print(f"No missing values found in column '{target_column}'")
            return None
        
        missing_row_idx = missing_rows.index[0]
        missing_row = test_df.iloc[missing_row_idx]
        
        print(f"Processing missing row {missing_row_idx}:")
        print(f"Target column '{target_column}': MISSING")
        
        # Show some context columns
        context_cols = [col for col in test_df.columns if not pd.isna(missing_row[col])][:10]
        print(f"Context columns (first 10 non-missing):")
        for col in context_cols:
            print(f"  {col}: {missing_row[col]}")
        
        # Run Gemini LLM1 imputation
        print(f"\nRunning Gemini LLM1 pipeline...")
        results = pipeline.run_gemini_pipeline(
            train_file=f"train_sets/{dataset_name}_train_original.csv",
            test_file=f"test_sets/{dataset_name}_test_MNAR.csv",
            dataset_name=dataset_name,
            missing_row_idx=missing_row_idx,
            target_column=target_column
        )
        
        if results:
            print(f"\n{'='*60}")
            print("IMPUTATION RESULTS")
            print(f"{'='*60}")
            
            imputed_value = results['llm1_prediction']
            reasoning = results['llm1_reasoning']
            confidence = results['confidence']
            
            print(f"Row {missing_row_idx} IMPUTATION:")
            print(f"  Before: MISSING")
            print(f"  After:  {imputed_value}")
            print(f"  Confidence: {confidence}")
            print(f"  Reasoning: {reasoning}")
            
            # Show before/after comparison
            print(f"\nBEFORE/AFTER COMPARISON:")
            print(f"  Original missing in '{target_column}': {test_df[target_column].isna().sum()}")
            
            # Create updated dataset
            test_df_updated = test_df.copy()
            test_df_updated.iloc[missing_row_idx, test_df_updated.columns.get_loc(target_column)] = imputed_value
            
            print(f"  After imputation: {test_df_updated[target_column].isna().sum()}")
            print(f"  Imputed value: {imputed_value}")
            
            # Save results
            output_file = f"clustering_results/{dataset_name}_test_with_imputation.csv"
            test_df_updated.to_csv(output_file, index=False)
            
            results_file = f"clustering_results/{dataset_name}_llm1_results.json"
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)
            
            print(f"\nFiles saved:")
            print(f"  Updated dataset: {output_file}")
            print(f"  Results: {results_file}")
            
            return results
        else:
            print("ERROR: Imputation failed")
            return None
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """
    Main function to run multi-dataset LLM1 imputation
    """
    print("MULTI-DATASET GEMINI LLM1 IMPUTATION")
    print("=" * 60)
    
    # Available datasets
    datasets = [
        {
            'name': 'phone',
            'test_file': 'test_sets/phone_test_MNAR.csv',
            'train_file': 'train_sets/phone_train_original.csv',
            'cluster_file': 'clustering_results/cluster_info_phone.json'
        },
        {
            'name': 'buy',
            'test_file': 'test_sets/buy_test_MNAR.csv',
            'train_file': 'train_sets/buy_train_original.csv',
            'cluster_file': 'clustering_results/cluster_info_buy.json'
        },
        {
            'name': 'restaurant',
            'test_file': 'test_sets/restaurant_test_MNAR.csv',
            'train_file': 'train_sets/restaurant_train_original.csv',
            'cluster_file': 'clustering_results/cluster_info_restaurant.json'
        },
        {
            'name': 'zomato',
            'test_file': 'test_sets/zomato_test_MNAR.csv',
            'train_file': 'train_sets/zomato_train_original.csv',
            'cluster_file': 'clustering_results/zomato_cluster_analysis.json'
        }
    ]
    
    # Check which datasets are available
    available_datasets = []
    for dataset in datasets:
        if (os.path.exists(dataset['test_file']) and 
            os.path.exists(dataset['train_file']) and 
            os.path.exists(dataset['cluster_file'])):
            available_datasets.append(dataset)
    
    if not available_datasets:
        print("ERROR: No complete datasets found!")
        return
    
    print(f"Available datasets: {len(available_datasets)}")
    for i, dataset in enumerate(available_datasets, 1):
        print(f"  {i}. {dataset['name'].upper()}")
    
    # Let user select dataset
    while True:
        try:
            choice = input(f"\nSelect dataset (1-{len(available_datasets)}): ").strip()
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(available_datasets):
                    selected_dataset = available_datasets[idx]
                    break
            print("Invalid choice. Please try again.")
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
            return
    
    # Analyze missing values
    missing_columns, test_df, train_df = analyze_dataset_missing_values(
        selected_dataset['name'],
        selected_dataset['test_file'],
        selected_dataset['train_file']
    )
    
    if not missing_columns:
        print("No missing columns found in this dataset!")
        return
    
    # Let user select column
    target_column = get_user_column_selection(missing_columns)
    if not target_column:
        return
    
    # Run LLM1 imputation
    results = run_llm1_imputation(
        selected_dataset['name'],
        test_df,
        train_df,
        target_column
    )
    
    if results:
        print(f"\nSUCCESS: LLM1 imputation completed!")
        print(f"Dataset: {selected_dataset['name']}")
        print(f"Column: {target_column}")
        print(f"Imputed value: {results['llm1_prediction']}")
        print(f"Confidence: {results['confidence']}")
    else:
        print("FAILED: LLM1 imputation failed!")

if __name__ == "__main__":
    main()














