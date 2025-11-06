"""
Test Groq LLM1 on buy_test_10percent_missing.csv
Compute MSE, SMAPE, KS Statistic evaluation metrics
"""

import pandas as pd
import numpy as np
import json
from scripts.groq_llm1_pipeline import GroqLLM1Pipeline
from scipy import stats
from datetime import datetime
import os

# Groq API Key
GROQ_API_KEY = "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"

def clean_price(price_str):
    """Convert price string to float"""
    if pd.isna(price_str) or price_str == '':
        return np.nan
    
    # Remove $, commas, and convert to float
    try:
        price_str = str(price_str).replace('$', '').replace(',', '').strip()
        return float(price_str)
    except:
        return np.nan

def calculate_mse(original_values, imputed_values):
    """Calculate Mean Squared Error for numerical values"""
    errors = []
    for orig, imp in zip(original_values, imputed_values):
        if pd.notna(orig) and pd.notna(imp):
            try:
                orig_num = float(clean_price(str(orig)) if '$' in str(orig) else orig)
                imp_num = float(clean_price(str(imp)) if '$' in str(imp) else imp)
                if not (np.isnan(orig_num) or np.isnan(imp_num)):
                    errors.append((orig_num - imp_num) ** 2)
            except:
                pass
    
    if errors:
        return np.mean(errors)
    return np.nan

def calculate_smape(original_values, imputed_values):
    """Calculate Symmetric Mean Absolute Percentage Error"""
    errors = []
    for orig, imp in zip(original_values, imputed_values):
        if pd.notna(orig) and pd.notna(imp):
            try:
                orig_num = float(clean_price(str(orig)) if '$' in str(orig) else orig)
                imp_num = float(clean_price(str(imp)) if '$' in str(imp) else imp)
                if not (np.isnan(orig_num) or np.isnan(imp_num)) and orig_num != 0:
                    numerator = abs(orig_num - imp_num)
                    denominator = (abs(orig_num) + abs(imp_num)) / 2
                    if denominator != 0:
                        errors.append(numerator / denominator)
            except:
                pass
    
    if errors:
        return np.mean(errors) * 100  # Return as percentage
    return np.nan

def calculate_ks_statistic(original_values, imputed_values):
    """Calculate Kolmogorov-Smirnov statistic"""
    orig_nums = []
    imp_nums = []
    
    for orig, imp in zip(original_values, imputed_values):
        if pd.notna(orig):
            try:
                orig_num = float(clean_price(str(orig)) if '$' in str(orig) else orig)
                if not np.isnan(orig_num):
                    orig_nums.append(orig_num)
            except:
                pass
        
        if pd.notna(imp):
            try:
                imp_num = float(clean_price(str(imp)) if '$' in str(imp) else imp)
                if not np.isnan(imp_num):
                    imp_nums.append(imp_num)
            except:
                pass
    
    if len(orig_nums) > 0 and len(imp_nums) > 0:
        ks_stat, p_value = stats.ks_2samp(orig_nums, imp_nums)
        return ks_stat, p_value
    return np.nan, np.nan

def evaluate_imputation_results(results, original_df, test_df):
    """
    Evaluate imputation results with MSE, SMAPE, KS Statistic
    """
    metrics = {
        'total_imputations': len(results),
        'successful_imputations': 0,
        'failed_imputations': 0,
        'accuracy': 0,
        'correct_matches': 0,
        'mse': np.nan,
        'smape': np.nan,
        'ks_statistic': np.nan,
        'ks_p_value': np.nan,
        'by_column': {}
    }
    
    successful_results = [r for r in results if r.get('success', False)]
    metrics['successful_imputations'] = len(successful_results)
    metrics['failed_imputations'] = len(results) - len(successful_results)
    
    # Group by column
    by_column = {}
    for result in successful_results:
        col = result['target_column']
        if col not in by_column:
            by_column[col] = {
                'original_values': [],
                'imputed_values': [],
                'matches': []
            }
        
        orig_val = result.get('original_value')
        imp_val = result.get('imputed_value')
        
        by_column[col]['original_values'].append(orig_val)
        by_column[col]['imputed_values'].append(imp_val)
        
        # Check exact match
        if pd.notna(orig_val) and pd.notna(imp_val):
            match = str(orig_val).lower().strip() == str(imp_val).lower().strip()
            by_column[col]['matches'].append(match)
            if match:
                metrics['correct_matches'] += 1
    
    # Calculate overall accuracy
    if successful_results:
        metrics['accuracy'] = (metrics['correct_matches'] / len(successful_results)) * 100
    
    # Calculate MSE, SMAPE, KS for all numerical values
    all_original = []
    all_imputed = []
    
    for result in successful_results:
        orig_val = result.get('original_value')
        imp_val = result.get('imputed_value')
        if pd.notna(orig_val) and pd.notna(imp_val):
            all_original.append(orig_val)
            all_imputed.append(imp_val)
    
    if all_original and all_imputed:
        metrics['mse'] = calculate_mse(all_original, all_imputed)
        metrics['smape'] = calculate_smape(all_original, all_imputed)
        metrics['ks_statistic'], metrics['ks_p_value'] = calculate_ks_statistic(all_original, all_imputed)
    
    # Calculate metrics by column
    for col, col_data in by_column.items():
        if col_data['original_values'] and col_data['imputed_values']:
            col_metrics = {
                'total': len(col_data['original_values']),
                'correct': sum(col_data['matches']) if col_data['matches'] else 0,
                'accuracy': (sum(col_data['matches']) / len(col_data['matches']) * 100) if col_data['matches'] else 0,
                'mse': calculate_mse(col_data['original_values'], col_data['imputed_values']),
                'smape': calculate_smape(col_data['original_values'], col_data['imputed_values']),
                'ks_statistic': calculate_ks_statistic(col_data['original_values'], col_data['imputed_values'])[0],
                'ks_p_value': calculate_ks_statistic(col_data['original_values'], col_data['imputed_values'])[1]
            }
            metrics['by_column'][col] = col_metrics
    
    return metrics

def test_buy_llm1_groq():
    """
    Test Groq LLM1 on buy_test_10percent_missing.csv
    """
    print("=" * 70)
    print("GROQ LLM1 TEST ON BUY DATASET (10% MISSING)")
    print("=" * 70)
    
    # Initialize pipeline
    pipeline = GroqLLM1Pipeline(GROQ_API_KEY)
    
    # Load datasets
    train_file = "train_sets/buy_train_original.csv"
    test_file = "test_sets_missing/buy_test_10percent_missing.csv"
    original_test_file = "test_sets/buy_test_original.csv"  # Ground truth test file
    
    if not os.path.exists(test_file):
        print(f"ERROR: Test file not found: {test_file}")
        return
    
    if not os.path.exists(original_test_file):
        print(f"ERROR: Original test file not found: {original_test_file}")
        return
    
    # Load test data with missing values
    test_df = pd.read_csv(test_file)
    
    # Load ground truth (original test data without missing values)
    original_df = pd.read_csv(original_test_file)
    
    # Verify they have same number of rows
    if len(test_df) != len(original_df):
        print(f"WARNING: Test data ({len(test_df)} rows) and original ({len(original_df)} rows) have different lengths")
        print("Using row-by-row matching by index")
    
    print(f"\nTest data: {len(test_df)} rows, {len(test_df.columns)} columns")
    print(f"Original data: {len(original_df)} rows, {len(original_df.columns)} columns")
    
    # Find all missing cells
    missing_cells = []
    for idx, row in test_df.iterrows():
        for col in test_df.columns:
            if pd.isna(row[col]) or str(row[col]).strip() == '':
                # Get original value from ground truth (same index)
                original_value = None
                if idx < len(original_df):
                    original_value = original_df.iloc[idx][col]
                missing_cells.append({
                    'row_idx': idx,
                    'column': col,
                    'original_value': original_value
                })
    
    print(f"\nTotal missing cells: {len(missing_cells)}")
    
    # Process each missing cell
    results = []
    successful_count = 0
    failed_count = 0
    
    for i, missing_cell in enumerate(missing_cells):
        row_idx = missing_cell['row_idx']
        col_name = missing_cell['column']
        original_value = missing_cell['original_value']
        
        print(f"\n{'='*70}")
        print(f"Processing {i+1}/{len(missing_cells)}: Row {row_idx}, Column '{col_name}'")
        print(f"{'='*70}")
        
        try:
            # Run Groq LLM1 pipeline
            result = pipeline.run_groq_pipeline(
                train_file=train_file,
                test_file=test_file,
                dataset_name="buy",
                missing_row_idx=row_idx,
                target_column=col_name
            )
            
            if result and result.get('llm1_prediction'):
                imputed_value = result['llm1_prediction']
                
                # Skip if prediction is NaN or empty
                if pd.isna(imputed_value) or str(imputed_value).strip().lower() in ['nan', 'none', 'null', '']:
                    print(f"  WARNING: LLM1 predicted NaN/empty - skipping")
                    results.append({
                        'row_idx': row_idx,
                        'target_column': col_name,
                        'original_value': original_value,
                        'imputed_value': None,
                        'cluster_id': result.get('cluster_id'),
                        'cluster_distance': result.get('cluster_distance', 0),
                        'success': False,
                        'error': 'Prediction was NaN or empty'
                    })
                    failed_count += 1
                    continue
                
                # Check if match
                match = False
                if pd.notna(original_value) and pd.notna(imputed_value):
                    match = str(original_value).lower().strip() == str(imputed_value).lower().strip()
                
                print(f"  Original: {original_value}")
                print(f"  Imputed: {imputed_value}")
                print(f"  Cluster: {result.get('cluster_id')} (distance: {result.get('cluster_distance', 0):.4f})")
                print(f"  Match: {match}")
                
                results.append({
                    'row_idx': row_idx,
                    'target_column': col_name,
                    'original_value': original_value,
                    'imputed_value': imputed_value,
                    'cluster_id': result.get('cluster_id'),
                    'cluster_distance': result.get('cluster_distance', 0),
                    'success': True,
                    'match': match
                })
                successful_count += 1
            else:
                print(f"  FAILED: {col_name}")
                results.append({
                    'row_idx': row_idx,
                    'target_column': col_name,
                    'original_value': original_value,
                    'imputed_value': None,
                    'success': False,
                    'error': 'No prediction returned'
                })
                failed_count += 1
                
        except Exception as e:
            error_msg = str(e).encode('ascii', errors='ignore').decode('ascii')
            print(f"  ERROR: {col_name} - {error_msg[:100]}")
            results.append({
                'row_idx': row_idx,
                'target_column': col_name,
                'original_value': original_value,
                'imputed_value': None,
                'success': False,
                'error': error_msg[:200]
            })
            failed_count += 1
    
    # Calculate evaluation metrics
    print(f"\n{'='*70}")
    print("EVALUATION METRICS")
    print(f"{'='*70}")
    
    metrics = evaluate_imputation_results(results, original_df, test_df)
    
    print(f"\nOverall Results:")
    print(f"  Total missing cells: {metrics['total_imputations']}")
    print(f"  Successful imputations: {metrics['successful_imputations']}")
    print(f"  Failed imputations: {metrics['failed_imputations']}")
    print(f"  Success rate: {(metrics['successful_imputations'] / metrics['total_imputations'] * 100):.1f}%")
    print(f"  Correct matches: {metrics['correct_matches']}/{metrics['successful_imputations']}")
    print(f"  Accuracy: {metrics['accuracy']:.2f}%")
    
    print(f"\nNumerical Metrics:")
    if pd.notna(metrics['mse']):
        print(f"  MSE (Mean Squared Error): {metrics['mse']:.4f}")
    if pd.notna(metrics['smape']):
        print(f"  SMAPE (Symmetric MAPE): {metrics['smape']:.2f}%")
    if pd.notna(metrics['ks_statistic']):
        print(f"  KS Statistic: {metrics['ks_statistic']:.4f}")
        print(f"  KS p-value: {metrics['ks_p_value']:.6f}")
    
    print(f"\nResults by Column:")
    for col, col_metrics in metrics['by_column'].items():
        print(f"\n  {col}:")
        print(f"    Total: {col_metrics['total']}")
        print(f"    Correct: {col_metrics['correct']}")
        print(f"    Accuracy: {col_metrics['accuracy']:.2f}%")
        if pd.notna(col_metrics['mse']):
            print(f"    MSE: {col_metrics['mse']:.4f}")
        if pd.notna(col_metrics['smape']):
            print(f"    SMAPE: {col_metrics['smape']:.2f}%")
        if pd.notna(col_metrics['ks_statistic']):
            print(f"    KS Statistic: {col_metrics['ks_statistic']:.4f}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"clustering_results/llm1_imputation/buy_10percent_groq_results_{timestamp}.json"
    metrics_file = f"clustering_results/llm1_imputation/buy_10percent_groq_metrics_{timestamp}.json"
    
    os.makedirs("clustering_results/llm1_imputation", exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    
    print(f"\n{'='*70}")
    print("RESULTS SAVED")
    print(f"{'='*70}")
    print(f"Results: {results_file}")
    print(f"Metrics: {metrics_file}")
    
    return results, metrics

if __name__ == "__main__":
    test_buy_llm1_groq()

