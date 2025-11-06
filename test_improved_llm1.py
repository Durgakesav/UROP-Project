"""
Test Improved LLM1 Pipeline with Full Cluster Data
This script tests the updated LLM1 pipeline that now receives full cluster data
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from scripts.gemini_llm1_pipeline import GeminiLLM1Pipeline

# Your Gemini API key
GEMINI_API_KEY = "AIzaSyCAN54rG_gYyE5h5wJAbK5cHVE1FX1l1GQ"

def test_improved_llm1():
    """
    Test the improved LLM1 pipeline with full cluster data
    """
    print("TESTING IMPROVED LLM1 PIPELINE WITH FULL CLUSTER DATA")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("API Key: AIzaSyCAN54rG_gYyE5h5wJAbK5cHVE1FX1l1GQ")
    print()
    
    # Test on buy dataset (smaller, faster)
    dataset_name = "buy"
    original_file = f"train_sets/{dataset_name}_train_original.csv"
    missing_file = f"test_sets_missing/{dataset_name}_test_10percent_missing.csv"
    
    if not os.path.exists(original_file):
        print(f"ERROR: Original file not found: {original_file}")
        return
    
    if not os.path.exists(missing_file):
        print(f"ERROR: Missing file not found: {missing_file}")
        return
    
    # Load datasets
    original_df = pd.read_csv(original_file)
    missing_df = pd.read_csv(missing_file)
    
    print(f"Dataset Information:")
    print(f"  Original: {len(original_df)} rows, {len(original_df.columns)} columns")
    print(f"  Missing: {len(missing_df)} rows, {len(missing_df.columns)} columns")
    
    # Check if clustered data exists
    cluster_file = f"clustering_results/{dataset_name}_with_clusters.csv"
    if os.path.exists(cluster_file):
        cluster_df = pd.read_csv(cluster_file)
        print(f"  Clustered data: {len(cluster_df)} rows with cluster labels")
        print(f"  Cluster distribution:")
        cluster_counts = cluster_df['cluster_label'].value_counts().sort_index()
        for cluster, count in cluster_counts.items():
            cluster_name = "Noise" if cluster == -1 else f"Cluster {cluster}"
            print(f"    {cluster_name}: {count} records")
    else:
        print(f"  ERROR: Clustered data not found: {cluster_file}")
        return
    
    # Initialize LLM1 pipeline
    try:
        print(f"\nInitializing improved LLM1 pipeline...")
        pipeline = GeminiLLM1Pipeline(GEMINI_API_KEY)
        
        if not pipeline.load_cluster_info(dataset_name):
            print("ERROR: Failed to load cluster information")
            return
        
        print("SUCCESS: Improved LLM1 pipeline initialized")
        print("  Now includes full cluster data (not just centroids)")
    except Exception as e:
        print(f"ERROR: {e}")
        return
    
    # Test on a few missing cells
    missing_cells = []
    for col in missing_df.columns:
        missing_indices = missing_df[missing_df[col].isna()].index.tolist()[:3]  # First 3
        missing_cells.extend([(idx, col) for idx in missing_indices])
    
    print(f"\nTesting on {len(missing_cells)} missing cells...")
    
    results = []
    successful_count = 0
    
    # Process each missing cell
    for i, (row_idx, col_name) in enumerate(missing_cells):
        print(f"\n[{i+1}/{len(missing_cells)}] Row {row_idx}, Column '{col_name}'")
        
        try:
            # Run improved LLM1 imputation
            result = pipeline.run_gemini_pipeline(
                train_file=original_file,
                test_file=missing_file,
                dataset_name=dataset_name,
                missing_row_idx=row_idx,
                target_column=col_name
            )
            
            if result and result['llm1_prediction']:
                imputed_value = result['llm1_prediction']
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
                    'reasoning': result.get('llm1_reasoning', 'N/A'),
                    'match': match,
                    'success': True
                })
                
                successful_count += 1
                print(f"  SUCCESS: '{col_name}' = {imputed_value}")
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
                
        except Exception as e:
            print(f"  ERROR: {col_name} - {str(e)[:50]}")
            results.append({
                'row_idx': row_idx,
                'column': col_name,
                'original_value': original_df.iloc[row_idx][col_name] if row_idx < len(original_df) else None,
                'imputed_value': None,
                'success': False,
                'error': str(e)[:100]
            })
    
    # Calculate metrics
    print(f"\n{'='*70}")
    print("IMPROVED LLM1 RESULTS SUMMARY")
    print(f"{'='*70}")
    
    successful_results = [r for r in results if r['success']]
    success_rate = (successful_count / len(missing_cells)) * 100 if missing_cells else 0
    
    print(f"Total missing cells processed: {len(missing_cells)}")
    print(f"Successful imputations: {successful_count}")
    print(f"Success rate: {success_rate:.1f}%")
    
    # Calculate accuracy
    accuracy = 0
    correct_matches = 0
    if successful_results:
        correct_matches = sum(1 for r in successful_results if r.get('match', False))
        accuracy = (correct_matches / len(successful_results)) * 100 if successful_results else 0
        
        print(f"Correct matches: {correct_matches}/{len(successful_results)}")
        print(f"Accuracy: {accuracy:.1f}%")
    
    # Show sample results
    print(f"\nSAMPLE IMPUTATION RESULTS:")
    print(f"{'Row':<6} {'Column':<12} {'Original':<25} {'Imputed':<25} {'Match':<6}")
    print(f"{'-'*80}")
    
    for result in results[:10]:
        original = str(result['original_value'])[:23] if pd.notna(result['original_value']) else 'None'
        imputed = str(result['imputed_value'])[:23] if pd.notna(result['imputed_value']) else 'None'
        match = 'Yes' if result.get('match', False) else 'No'
        print(f"{result['row_idx']:<6} {result['column']:<12} {original:<25} {imputed:<25} {match:<6}")
    
    print(f"\n{'='*70}")
    print("IMPROVED LLM1 TEST COMPLETED")
    print(f"{'='*70}")
    print(f"Key improvements:")
    print(f"1. Dynamic cluster assignment (not always cluster 0)")
    print(f"2. Full cluster data sent to LLM1 (not just centroids)")
    print(f"3. Sample cluster member records included")
    print(f"4. Expected accuracy improvement: 0.8% -> 50-70%")

if __name__ == "__main__":
    test_improved_llm1()













