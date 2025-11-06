"""
Automated Multi-Dataset LLM1 Imputation Analysis
This script automatically tests all datasets and shows missing columns
"""

import pandas as pd
import json
import os
from scripts.gemini_llm1_pipeline import GeminiLLM1Pipeline

# Your Gemini API key
GEMINI_API_KEY = "AIzaSyC6yMHoOO60kywdWgBWcwUtB2nbTo8_kAg"

def analyze_all_datasets():
    """
    Analyze all available datasets and their missing columns
    """
    print("AUTOMATED MULTI-DATASET ANALYSIS")
    print("=" * 70)
    
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
    
    results = {}
    
    for dataset in datasets:
        print(f"\n{'='*70}")
        print(f"ANALYZING DATASET: {dataset['name'].upper()}")
        print(f"{'='*70}")
        
        try:
            # Check if files exist
            if not (os.path.exists(dataset['test_file']) and 
                   os.path.exists(dataset['train_file']) and 
                   os.path.exists(dataset['cluster_file'])):
                print(f"SKIPPING: Missing required files for {dataset['name']}")
                continue
            
            # Load datasets
            test_df = pd.read_csv(dataset['test_file'])
            train_df = pd.read_csv(dataset['train_file'])
            
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
            
            results[dataset['name']] = {
                'missing_columns': missing_columns,
                'test_df': test_df,
                'train_df': train_df,
                'total_missing': len(missing_columns)
            }
            
        except Exception as e:
            print(f"ERROR: Could not analyze dataset {dataset['name']}: {e}")
            results[dataset['name']] = {'error': str(e)}
    
    return results

def run_llm1_imputation_auto(dataset_name, test_df, train_df, target_column):
    """
    Run Gemini LLM1 imputation automatically
    """
    print(f"\n{'='*70}")
    print(f"RUNNING GEMINI LLM1 IMPUTATION")
    print(f"Dataset: {dataset_name.upper()}")
    print(f"Target Column: {target_column}")
    print(f"{'='*70}")
    
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
        context_cols = [col for col in test_df.columns if not pd.isna(missing_row[col])][:5]
        print(f"Context columns (first 5 non-missing):")
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
            print(f"\nIMPUTATION RESULTS:")
            print(f"  Before: MISSING")
            print(f"  After:  {results['llm1_prediction']}")
            print(f"  Confidence: {results['confidence']}")
            print(f"  Reasoning: {results['llm1_reasoning'][:100]}...")
            
            return results
        else:
            print("ERROR: Imputation failed")
            return None
            
    except Exception as e:
        print(f"ERROR: {e}")
        return None

def main():
    """
    Main function to run automated analysis
    """
    print("AUTOMATED MULTI-DATASET GEMINI LLM1 ANALYSIS")
    print("=" * 70)
    
    # Analyze all datasets
    results = analyze_all_datasets()
    
    # Summary of missing columns
    print(f"\n{'='*70}")
    print("SUMMARY OF MISSING COLUMNS")
    print(f"{'='*70}")
    
    for dataset_name, data in results.items():
        if 'error' in data:
            print(f"{dataset_name.upper()}: ERROR - {data['error']}")
        else:
            missing_cols = data['missing_columns']
            print(f"{dataset_name.upper()}: {len(missing_cols)} columns with missing values")
            for col_info in missing_cols[:3]:  # Show first 3
                print(f"  - {col_info['column']}: {col_info['missing_count']} missing ({col_info['percentage']:.1f}%)")
            if len(missing_cols) > 3:
                print(f"  ... and {len(missing_cols) - 3} more columns")
    
    # Test LLM1 imputation on each dataset
    print(f"\n{'='*70}")
    print("TESTING LLM1 IMPUTATION ON EACH DATASET")
    print(f"{'='*70}")
    
    for dataset_name, data in results.items():
        if 'error' in data or not data.get('missing_columns'):
            continue
        
        # Select the column with most missing values for testing
        missing_columns = data['missing_columns']
        target_column = max(missing_columns, key=lambda x: x['missing_count'])['column']
        
        print(f"\nTesting {dataset_name.upper()} - Column: {target_column}")
        
        imputation_result = run_llm1_imputation_auto(
            dataset_name,
            data['test_df'],
            data['train_df'],
            target_column
        )
        
        if imputation_result:
            print(f"SUCCESS: {dataset_name.upper()} - {target_column} = {imputation_result['llm1_prediction']}")
        else:
            print(f"FAILED: {dataset_name.upper()} - {target_column}")

if __name__ == "__main__":
    main()














