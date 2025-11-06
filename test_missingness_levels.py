"""
Test LLM1 Imputation on Different Missingness Levels
This script tests Gemini LLM1 on the newly created missing datasets
"""

import pandas as pd
import json
import os
from scripts.gemini_llm1_pipeline import GeminiLLM1Pipeline

# Your Gemini API key
GEMINI_API_KEY = "AIzaSyC6yMHoOO60kywdWgBWcwUtB2nbTo8_kAg"

def test_missingness_levels():
    """
    Test LLM1 imputation on different missingness levels
    """
    print("TESTING LLM1 IMPUTATION ON DIFFERENT MISSINGNESS LEVELS")
    print("=" * 80)
    
    # Define test datasets
    test_files = [
        {
            'name': 'phone',
            'files': [
                'test_sets_missing/phone_test_10percent_missing.csv',
                'test_sets_missing/phone_test_30percent_missing.csv',
                'test_sets_missing/phone_test_50percent_missing.csv'
            ],
            'train_file': 'train_sets/phone_train_original.csv',
            'cluster_file': 'clustering_results/cluster_info_phone.json',
            'target_columns': ['brand', 'model', 'NFC', 'price']
        },
        {
            'name': 'buy',
            'files': [
                'test_sets_missing/buy_test_10percent_missing.csv',
                'test_sets_missing/buy_test_30percent_missing.csv',
                'test_sets_missing/buy_test_50percent_missing.csv'
            ],
            'train_file': 'train_sets/buy_train_original.csv',
            'cluster_file': 'clustering_results/cluster_info_buy.json',
            'target_columns': ['name', 'manufacturer', 'price']
        },
        {
            'name': 'restaurant',
            'files': [
                'test_sets_missing/restaurant_test_10percent_missing.csv',
                'test_sets_missing/restaurant_test_30percent_missing.csv',
                'test_sets_missing/restaurant_test_50percent_missing.csv'
            ],
            'train_file': 'train_sets/restaurant_train_original.csv',
            'cluster_file': 'clustering_results/cluster_info_restaurant.json',
            'target_columns': ['name', 'type', 'city']
        },
        {
            'name': 'zomato',
            'files': [
                'test_sets_missing/zomato_test_10percent_missing.csv',
                'test_sets_missing/zomato_test_30percent_missing.csv',
                'test_sets_missing/zomato_test_50percent_missing.csv'
            ],
            'train_file': 'train_sets/zomato_train_original.csv',
            'cluster_file': 'clustering_results/zomato_cluster_analysis.json',
            'target_columns': ['restaurant_name', 'cuisine', 'area']
        }
    ]
    
    results = {}
    
    for dataset in test_files:
        print(f"\n{'='*80}")
        print(f"TESTING DATASET: {dataset['name'].upper()}")
        print(f"{'='*80}")
        
        dataset_results = {}
        
        for test_file in dataset['files']:
            if not os.path.exists(test_file):
                print(f"SKIP: {test_file} not found")
                continue
            
            # Extract missingness level from filename
            missingness = test_file.split('_')[-2]  # e.g., "10percent"
            missingness_pct = missingness.replace('percent', '')
            
            print(f"\nTesting {missingness_pct}% missingness...")
            
            try:
                # Load test data
                test_df = pd.read_csv(test_file)
                
                # Find columns with missing values
                missing_columns = []
                for col in test_df.columns:
                    missing_count = test_df[col].isna().sum()
                    if missing_count > 0:
                        missing_columns.append({
                            'column': col,
                            'missing_count': missing_count,
                            'percentage': (missing_count / len(test_df)) * 100
                        })
                
                print(f"  Missing columns: {len(missing_columns)}")
                
                # Test LLM1 on first few missing columns
                test_results = []
                for col_info in missing_columns[:2]:  # Test first 2 columns
                    col_name = col_info['column']
                    if col_name in dataset['target_columns']:
                        print(f"  Testing column: {col_name}")
                        
                        # Find first missing row
                        missing_rows = test_df[test_df[col_name].isna()]
                        if len(missing_rows) > 0:
                            missing_row_idx = missing_rows.index[0]
                            
                            # Run LLM1 imputation
                            try:
                                pipeline = GeminiLLM1Pipeline(GEMINI_API_KEY)
                                result = pipeline.run_gemini_pipeline(
                                    train_file=dataset['train_file'],
                                    test_file=test_file,
                                    dataset_name=dataset['name'],
                                    missing_row_idx=missing_row_idx,
                                    target_column=col_name
                                )
                                
                                if result:
                                    test_results.append({
                                        'column': col_name,
                                        'prediction': result['llm1_prediction'],
                                        'confidence': result['confidence'],
                                        'success': True
                                    })
                                    print(f"    SUCCESS: {col_name} = {result['llm1_prediction']}")
                                else:
                                    test_results.append({
                                        'column': col_name,
                                        'success': False,
                                        'error': 'Imputation failed'
                                    })
                                    print(f"    FAILED: {col_name}")
                            except Exception as e:
                                test_results.append({
                                    'column': col_name,
                                    'success': False,
                                    'error': str(e)
                                })
                                print(f"    ERROR: {col_name} - {e}")
                
                dataset_results[missingness_pct] = {
                    'missing_columns': len(missing_columns),
                    'test_results': test_results,
                    'success_rate': sum(1 for r in test_results if r.get('success', False)) / len(test_results) if test_results else 0
                }
                
            except Exception as e:
                print(f"ERROR processing {test_file}: {e}")
                dataset_results[missingness_pct] = {'error': str(e)}
        
        results[dataset['name']] = dataset_results
    
    return results

def print_summary(results):
    """
    Print summary of test results
    """
    print(f"\n{'='*80}")
    print("SUMMARY OF LLM1 IMPUTATION TESTS")
    print(f"{'='*80}")
    
    for dataset_name, dataset_results in results.items():
        print(f"\n{dataset_name.upper()}:")
        print(f"{'Missingness':<15} {'Columns':<10} {'Success Rate':<15} {'Status'}")
        print(f"{'-'*60}")
        
        for missingness, data in dataset_results.items():
            if 'error' in data:
                print(f"{missingness}%{'':<10} {'ERROR':<10} {'0%':<15} {data['error']}")
            else:
                success_rate = f"{data['success_rate']*100:.1f}%"
                status = "SUCCESS" if data['success_rate'] > 0.5 else "PARTIAL" if data['success_rate'] > 0 else "FAILED"
                print(f"{missingness}%{'':<10} {data['missing_columns']:<10} {success_rate:<15} {status}")

def main():
    """
    Main function
    """
    print("LLM1 IMPUTATION TESTING ON MISSINGNESS LEVELS")
    print("=" * 80)
    print("Testing Gemini LLM1 on 10%, 30%, and 50% missing datasets")
    print()
    
    # Run tests
    results = test_missingness_levels()
    
    # Print summary
    print_summary(results)
    
    # Save results
    with open("clustering_results/missingness_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to: clustering_results/missingness_test_results.json")

if __name__ == "__main__":
    main()















