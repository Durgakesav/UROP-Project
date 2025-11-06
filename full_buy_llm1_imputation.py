"""
Full LLM1 Clustering Imputation for Buy Dataset
This script imputes ALL missing values in buy_test_10percent_missing.csv using LLM1 clustering
"""

import pandas as pd
import numpy as np
import json
import os
from sklearn.metrics import mean_squared_error, accuracy_score
from scripts.gemini_llm1_pipeline import GeminiLLM1Pipeline

# Your new Gemini API key
GEMINI_API_KEY = "AIzaSyCAN54rG_gYyE5h5wJAbK5cHVE1FX1l1GQ"

def full_buy_llm1_imputation():
    """
    Full LLM1 clustering imputation for buy dataset
    """
    print("FULL LLM1 CLUSTERING IMPUTATION - BUY DATASET")
    print("=" * 70)
    
    # Load datasets
    original_file = "train_sets/buy_train_original.csv"
    missing_file = "test_sets_missing/buy_test_10percent_missing.csv"
    
    if not os.path.exists(original_file) or not os.path.exists(missing_file):
        print("ERROR: Required files not found")
        return
    
    original_df = pd.read_csv(original_file)
    missing_df = pd.read_csv(missing_file)
    
    print(f"Original dataset: {len(original_df)} rows, {len(original_df.columns)} columns")
    print(f"Missing dataset: {len(missing_df)} rows, {len(missing_df.columns)} columns")
    
    # Find ALL missing cells
    missing_cells = []
    for col in missing_df.columns:
        missing_indices = missing_df[missing_df[col].isna()].index.tolist()
        missing_cells.extend([(idx, col) for idx in missing_indices])
    
    print(f"Total missing cells to impute: {len(missing_cells)}")
    
    # Show missing cells by column
    print("\nMissing cells by column:")
    for col in missing_df.columns:
        missing_count = missing_df[col].isna().sum()
        if missing_count > 0:
            print(f"  {col}: {missing_count} missing")
    
    # Initialize pipeline with new API key
    try:
        pipeline = GeminiLLM1Pipeline(GEMINI_API_KEY)
        if not pipeline.load_cluster_info("buy"):
            print("ERROR: Failed to load cluster information")
            return
    except Exception as e:
        print(f"ERROR: {e}")
        return
    
    # Process ALL missing cells
    print(f"\nProcessing {len(missing_cells)} missing cells with LLM1 clustering...")
    
    imputed_df = missing_df.copy()
    results = []
    successful_count = 0
    failed_count = 0
    
    for i, (row_idx, col_name) in enumerate(missing_cells):
        print(f"\nProcessing {i+1}/{len(missing_cells)}: Row {row_idx}, Column '{col_name}'")
        
        try:
            # Run LLM1 imputation
            result = pipeline.run_gemini_pipeline(
                train_file=original_file,
                test_file=missing_file,
                dataset_name="buy",
                missing_row_idx=row_idx,
                target_column=col_name
            )
            
            if result and result['llm1_prediction']:
                imputed_value = result['llm1_prediction']
                imputed_df.iloc[row_idx, imputed_df.columns.get_loc(col_name)] = imputed_value
                
                original_value = original_df.iloc[row_idx][col_name] if row_idx < len(original_df) else None
                
                # Check if prediction matches original
                match = False
                if pd.notna(original_value) and pd.notna(imputed_value):
                    match = str(original_value).lower() == str(imputed_value).lower()
                
                results.append({
                    'row_idx': row_idx,
                    'column': col_name,
                    'original_value': original_value,
                    'imputed_value': imputed_value,
                    'confidence': result['confidence'],
                    'reasoning': result['llm1_reasoning'],
                    'match': match,
                    'success': True
                })
                
                successful_count += 1
                print(f"  SUCCESS: {col_name} = {imputed_value}")
                print(f"  Original: {original_value}")
                print(f"  Match: {match}")
            else:
                print(f"  FAILED: {col_name}")
                results.append({
                    'row_idx': row_idx,
                    'column': col_name,
                    'original_value': original_df.iloc[row_idx][col_name] if row_idx < len(original_df) else None,
                    'imputed_value': None,
                    'success': False,
                    'error': 'No prediction returned'
                })
                failed_count += 1
                
        except Exception as e:
            print(f"  ERROR: {col_name} - {e}")
            results.append({
                'row_idx': row_idx,
                'column': col_name,
                'original_value': original_df.iloc[row_idx][col_name] if row_idx < len(original_df) else None,
                'imputed_value': None,
                'success': False,
                'error': str(e)
            })
            failed_count += 1
    
    # Calculate comprehensive metrics
    print(f"\n{'='*70}")
    print("COMPREHENSIVE RESULTS")
    print(f"{'='*70}")
    
    successful_results = [r for r in results if r['success']]
    print(f"Total missing cells: {len(missing_cells)}")
    print(f"Successful imputations: {successful_count}")
    print(f"Failed imputations: {failed_count}")
    print(f"Success rate: {successful_count/len(missing_cells)*100:.1f}%")
    
    if successful_results:
        # Calculate accuracy
        correct_matches = sum(1 for r in successful_results if r.get('match', False))
        accuracy = correct_matches / len(successful_results) * 100 if successful_results else 0
        
        print(f"Correct matches: {correct_matches}/{len(successful_results)}")
        print(f"Accuracy: {accuracy:.1f}%")
        
        # Show results by column
        print(f"\nResults by column:")
        for col in missing_df.columns:
            col_results = [r for r in successful_results if r['column'] == col]
            if col_results:
                col_correct = sum(1 for r in col_results if r.get('match', False))
                col_accuracy = col_correct / len(col_results) * 100 if col_results else 0
                print(f"  {col}: {len(col_results)} imputed, {col_correct} correct ({col_accuracy:.1f}%)")
    
    # Save all results
    os.makedirs("clustering_results/llm1_imputation", exist_ok=True)
    
    # Save imputed dataset
    imputed_file = "clustering_results/llm1_imputation/buy_full_imputed_llm1.csv"
    imputed_df.to_csv(imputed_file, index=False)
    print(f"\nImputed dataset saved: {imputed_file}")
    
    # Save detailed results
    results_file = "clustering_results/llm1_imputation/buy_full_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Detailed results saved: {results_file}")
    
    # Save summary
    summary = {
        'dataset': 'buy',
        'total_missing_cells': len(missing_cells),
        'successful_imputations': successful_count,
        'failed_imputations': failed_count,
        'success_rate': successful_count/len(missing_cells)*100,
        'accuracy': accuracy if successful_results else 0,
        'correct_matches': correct_matches if successful_results else 0,
        'total_successful': len(successful_results)
    }
    
    summary_file = "clustering_results/llm1_imputation/buy_full_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved: {summary_file}")
    
    # Show sample results
    print(f"\nSAMPLE IMPUTATION RESULTS:")
    print(f"{'Row':<5} {'Column':<12} {'Original':<20} {'Imputed':<20} {'Match':<6}")
    print(f"{'-'*70}")
    
    for result in results[:10]:  # Show first 10 results
        original = str(result['original_value'])[:18] if pd.notna(result['original_value']) else 'None'
        imputed = str(result['imputed_value'])[:18] if pd.notna(result['imputed_value']) else 'None'
        match = 'Yes' if result.get('match', False) else 'No'
        print(f"{result['row_idx']:<5} {result['column']:<12} {original:<20} {imputed:<20} {match:<6}")
    
    if len(results) > 10:
        print(f"... and {len(results) - 10} more results")

if __name__ == "__main__":
    full_buy_llm1_imputation()














