"""
Fast LLM1 Clustering Imputation for Restaurant Dataset - 10% Missing
Minimal API usage with cluster-specific data only
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from scripts.gemini_llm1_pipeline import GeminiLLM1Pipeline

# Your Gemini API key
GEMINI_API_KEY = "AIzaSyCAN54rG_gYyE5h5wJAbK5cHVE1FX1l1GQ"

def fast_restaurant_llm1():
    """
    Fast LLM1 clustering on restaurant 10% missing data
    Uses only cluster-specific data, minimal API calls
    """
    print("FAST LLM1 CLUSTERING - RESTAURANT 10% MISSING")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("API Key: AIzaSyCAN54rG_gYyE5h5wJAbK5cHVE1FX1l1GQ")
    print()
    
    # Load datasets
    original_file = "train_sets/restaurant_train_original.csv"
    missing_file = "test_sets_missing/restaurant_test_10percent_missing.csv"
    
    if not os.path.exists(original_file):
        print("ERROR: Original file not found")
        return
    
    if not os.path.exists(missing_file):
        print("ERROR: Missing file not found")
        return
    
    # Load datasets
    original_df = pd.read_csv(original_file)
    missing_df = pd.read_csv(missing_file)
    
    print(f"Dataset Information:")
    print(f"  Original: {len(original_df)} rows, {len(original_df.columns)} columns")
    print(f"  Missing: {len(missing_df)} rows, {len(missing_df.columns)} columns")
    
    # Check cluster info
    cluster_file = "clustering_results/cluster_info_restaurant.json"
    if os.path.exists(cluster_file):
        with open(cluster_file, 'r') as f:
            cluster_info = json.load(f)
        
        n_clusters = cluster_info.get('n_clusters', 0)
        print(f"  Clusters available: {n_clusters}")
        
        if n_clusters == 0:
            print("  WARNING: No clusters found - using full dataset fallback")
            print("  This will use more API calls but still minimal approach")
    else:
        print("  WARNING: No cluster info found - using full dataset fallback")
        n_clusters = 0
    
    # Find missing cells (sample only first 5 for speed)
    missing_cells = []
    for col in missing_df.columns:
        missing_indices = missing_df[missing_df[col].isna()].index.tolist()[:5]  # Only first 5
        missing_cells.extend([(idx, col) for idx in missing_indices])
    
    print(f"  Missing cells to process: {len(missing_cells)} (sampled for speed)")
    
    # Initialize LLM1 pipeline
    try:
        print(f"\nInitializing LLM1 pipeline...")
        pipeline = GeminiLLM1Pipeline(GEMINI_API_KEY)
        
        if n_clusters > 0:
            if not pipeline.load_cluster_info("restaurant"):
                print("ERROR: Failed to load cluster information")
                return
        else:
            print("Using full dataset fallback (no clusters available)")
        
        print("SUCCESS: LLM1 pipeline initialized")
    except Exception as e:
        print(f"ERROR: {e}")
        return
    
    # Process missing cells (minimal approach)
    print(f"\nProcessing {len(missing_cells)} missing cells (minimal API usage)...")
    
    imputed_df = missing_df.copy()
    results = []
    successful_count = 0
    failed_count = 0
    
    # Process each missing cell
    for i, (row_idx, col_name) in enumerate(missing_cells):
        print(f"\nProcessing {i+1}/{len(missing_cells)}: Row {row_idx}, Column '{col_name}'")
        
        try:
            # Run LLM1 imputation
            result = pipeline.run_gemini_pipeline(
                train_file=original_file,
                test_file=missing_file,
                dataset_name="restaurant",
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
    
    # Calculate metrics
    print(f"\n{'='*60}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*60}")
    
    successful_results = [r for r in results if r['success']]
    success_rate = (successful_count / len(missing_cells)) * 100 if missing_cells else 0
    
    print(f"Total missing cells processed: {len(missing_cells)}")
    print(f"Successful imputations: {successful_count}")
    print(f"Failed imputations: {failed_count}")
    print(f"Success rate: {success_rate:.1f}%")
    
    # Calculate accuracy
    accuracy = 0
    correct_matches = 0
    if successful_results:
        correct_matches = sum(1 for r in successful_results if r.get('match', False))
        accuracy = (correct_matches / len(successful_results)) * 100 if successful_results else 0
        
        print(f"Correct matches: {correct_matches}/{len(successful_results)}")
        print(f"Accuracy: {accuracy:.1f}%")
    
    # Results by column
    print(f"\nResults by column:")
    for col in missing_df.columns:
        col_results = [r for r in successful_results if r['column'] == col]
        if col_results:
            col_correct = sum(1 for r in col_results if r.get('match', False))
            col_accuracy = (col_correct / len(col_results)) * 100 if col_results else 0
            print(f"  {col}: {len(col_results)} imputed, {col_correct} correct ({col_accuracy:.1f}%)")
    
    # Create output directory
    output_dir = "clustering_results/llm1_imputation"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save imputed dataset
    imputed_file = f"{output_dir}/restaurant_10percent_imputed_llm1.csv"
    imputed_df.to_csv(imputed_file, index=False)
    print(f"\nImputed dataset saved: {imputed_file}")
    
    # Save detailed results
    results_file = f"{output_dir}/restaurant_10percent_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Detailed results saved: {results_file}")
    
    # Save summary
    summary = {
        'dataset': 'restaurant',
        'missingness': '10%',
        'timestamp': datetime.now().isoformat(),
        'api_key': GEMINI_API_KEY[:20] + '...',
        'total_missing_cells_processed': len(missing_cells),
        'successful_imputations': successful_count,
        'failed_imputations': failed_count,
        'success_rate': success_rate,
        'accuracy': accuracy,
        'correct_matches': correct_matches,
        'total_successful': len(successful_results),
        'clusters_available': n_clusters,
        'method': 'LLM1 Clustering (Minimal API)'
    }
    
    summary_file = f"{output_dir}/restaurant_10percent_summary.json"
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
    
    print(f"\n{'='*60}")
    print("FAST LLM1 IMPUTATION COMPLETED")
    print(f"{'='*60}")
    print(f"API calls used: {len(missing_cells)} (minimal)")
    print(f"All results saved to: {output_dir}/")

if __name__ == "__main__":
    fast_restaurant_llm1()














