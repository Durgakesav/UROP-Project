"""
Fast LLM1 Imputation for Buy Dataset
This script quickly imputes missing values using LLM1 clustering on the smaller buy dataset
"""

import pandas as pd
import numpy as np
import json
import os
from sklearn.metrics import mean_squared_error, accuracy_score
from scripts.gemini_llm1_pipeline import GeminiLLM1Pipeline

# Your Gemini API key
GEMINI_API_KEY = "AIzaSyC6yMHoOO60kywdWgBWcwUtB2nbTo8_kAg"

def fast_imputation_buy():
    """
    Fast LLM1 imputation for buy dataset
    """
    print("FAST LLM1 IMPUTATION FOR BUY DATASET")
    print("=" * 60)
    
    # Load datasets
    original_file = "train_sets/buy_train_original.csv"
    missing_file = "test_sets_missing/buy_test_10percent_missing.csv"
    
    if not os.path.exists(original_file) or not os.path.exists(missing_file):
        print("ERROR: Required files not found")
        return
    
    original_df = pd.read_csv(original_file)
    missing_df = pd.read_csv(missing_file)
    
    print(f"Original: {len(original_df)} rows, {len(original_df.columns)} columns")
    print(f"Missing: {len(missing_df)} rows, {len(missing_df.columns)} columns")
    
    # Find missing cells
    missing_cells = []
    for col in missing_df.columns:
        missing_indices = missing_df[missing_df[col].isna()].index.tolist()
        missing_cells.extend([(idx, col) for idx in missing_indices])
    
    print(f"Missing cells to impute: {len(missing_cells)}")
    
    # Initialize pipeline
    try:
        pipeline = GeminiLLM1Pipeline(GEMINI_API_KEY)
        if not pipeline.load_cluster_info("buy"):
            print("ERROR: Failed to load cluster information")
            return
    except Exception as e:
        print(f"ERROR: {e}")
        return
    
    # Process only first 5 missing cells for speed
    test_cells = missing_cells[:5]
    print(f"Testing on first {len(test_cells)} missing cells for speed")
    
    imputed_df = missing_df.copy()
    results = []
    
    for i, (row_idx, col_name) in enumerate(test_cells):
        print(f"\nProcessing {i+1}/{len(test_cells)}: Row {row_idx}, Column '{col_name}'")
        
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
                
                results.append({
                    'row_idx': row_idx,
                    'column': col_name,
                    'original_value': original_value,
                    'imputed_value': imputed_value,
                    'confidence': result['confidence'],
                    'success': True
                })
                
                print(f"  SUCCESS: {col_name} = {imputed_value}")
                print(f"  Original: {original_value}")
                print(f"  Match: {str(original_value).lower() == str(imputed_value).lower()}")
            else:
                print(f"  FAILED: {col_name}")
                results.append({
                    'row_idx': row_idx,
                    'column': col_name,
                    'original_value': original_df.iloc[row_idx][col_name] if row_idx < len(original_df) else None,
                    'imputed_value': None,
                    'success': False
                })
                
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
    
    # Calculate metrics
    successful = [r for r in results if r['success']]
    print(f"\nRESULTS SUMMARY:")
    print(f"Total tested: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Success rate: {len(successful)/len(results)*100:.1f}%")
    
    if successful:
        correct = 0
        for result in successful:
            if pd.notna(result['original_value']) and pd.notna(result['imputed_value']):
                if str(result['original_value']).lower() == str(result['imputed_value']).lower():
                    correct += 1
        
        accuracy = correct / len(successful) if successful else 0
        print(f"Accuracy: {accuracy:.4f} ({correct}/{len(successful)})")
    
    # Save results
    os.makedirs("clustering_results/llm1_imputation", exist_ok=True)
    
    # Save imputed dataset
    imputed_df.to_csv("clustering_results/llm1_imputation/buy_imputed_llm1.csv", index=False)
    print(f"\nImputed dataset saved: clustering_results/llm1_imputation/buy_imputed_llm1.csv")
    
    # Save results
    with open("clustering_results/llm1_imputation/buy_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: clustering_results/llm1_imputation/buy_results.json")
    
    # Show detailed results
    print(f"\nDETAILED RESULTS:")
    for result in results:
        print(f"Row {result['row_idx']}, {result['column']}:")
        print(f"  Original: {result['original_value']}")
        print(f"  Imputed: {result['imputed_value']}")
        print(f"  Success: {result['success']}")
        if result['success'] and pd.notna(result['original_value']):
            match = str(result['original_value']).lower() == str(result['imputed_value']).lower()
            print(f"  Match: {match}")
        print()

if __name__ == "__main__":
    fast_imputation_buy()














