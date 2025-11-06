"""
LLM1 Clustering-based Imputation for Phone Dataset
This script imputes all missing values using LLM1 with cluster-specific data only
"""

import pandas as pd
import numpy as np
import json
import os
from sklearn.metrics import mean_squared_error, accuracy_score
from scripts.gemini_llm1_pipeline import GeminiLLM1Pipeline

# Your Gemini API key
GEMINI_API_KEY = "AIzaSyC6yMHoOO60kywdWgBWcwUtB2nbTo8_kAg"

def load_datasets():
    """
    Load the phone datasets
    """
    print("LOADING PHONE DATASETS")
    print("=" * 60)
    
    # Load datasets
    original_file = "train_sets/phone_train_original.csv"
    missing_file = "test_sets_missing/phone_test_10percent_missing.csv"
    
    if not os.path.exists(original_file) or not os.path.exists(missing_file):
        print("ERROR: Required files not found")
        return None, None, None
    
    # Load datasets
    original_df = pd.read_csv(original_file)
    missing_df = pd.read_csv(missing_file)
    
    print(f"Original dataset: {len(original_df)} rows, {len(original_df.columns)} columns")
    print(f"Missing dataset: {len(missing_df)} rows, {len(missing_df.columns)} columns")
    
    # Calculate missing statistics
    missing_cells = missing_df.isna().sum().sum()
    total_cells = len(missing_df) * len(missing_df.columns)
    missing_percentage = (missing_cells / total_cells) * 100
    
    print(f"Missing cells: {missing_cells}/{total_cells} ({missing_percentage:.1f}%)")
    
    return original_df, missing_df, missing_cells

def find_missing_cells(missing_df):
    """
    Find all missing cells and their positions
    """
    print("\nFINDING MISSING CELLS")
    print("=" * 60)
    
    missing_cells = []
    
    for col in missing_df.columns:
        missing_indices = missing_df[missing_df[col].isna()].index.tolist()
        if missing_indices:
            missing_cells.extend([(idx, col) for idx in missing_indices])
    
    print(f"Total missing cells to impute: {len(missing_cells)}")
    
    # Show first 10 missing cells
    print("\nFirst 10 missing cells:")
    for i, (row_idx, col_name) in enumerate(missing_cells[:10]):
        print(f"  {i+1}. Row {row_idx}, Column '{col_name}'")
    
    return missing_cells

def assign_cluster_to_missing_row(missing_row, centroids):
    """
    Assign cluster to a missing row based on available features
    """
    # For simplicity, assign to cluster 0 (first cluster)
    # In practice, you would calculate distance to centroids
    return "0"

def impute_missing_cells(missing_df, missing_cells, original_df):
    """
    Impute all missing cells using LLM1 clustering
    """
    print("\nIMPUTING MISSING CELLS WITH LLM1")
    print("=" * 60)
    
    try:
        # Initialize pipeline
        pipeline = GeminiLLM1Pipeline(GEMINI_API_KEY)
        
        # Load cluster info
        if not pipeline.load_cluster_info("phone"):
            print("ERROR: Failed to load cluster information")
            return None
        
        # Create results dataframe
        imputed_df = missing_df.copy()
        imputation_results = []
        
        print(f"Processing {len(missing_cells)} missing cells...")
        
        # Process each missing cell
        for i, (row_idx, col_name) in enumerate(missing_cells):
            print(f"\nProcessing cell {i+1}/{len(missing_cells)}: Row {row_idx}, Column '{col_name}'")
            
            try:
                # Get the missing row
                missing_row = missing_df.iloc[row_idx]
                
                # Assign cluster (simplified - use cluster 0)
                cluster_id = assign_cluster_to_missing_row(missing_row, pipeline.centroids)
                
                # Run LLM1 imputation
                result = pipeline.run_gemini_pipeline(
                    train_file="train_sets/phone_train_original.csv",
                    test_file="test_sets_missing/phone_test_10percent_missing.csv",
                    dataset_name="phone",
                    missing_row_idx=row_idx,
                    target_column=col_name
                )
                
                if result and result['llm1_prediction']:
                    # Update the imputed dataframe
                    imputed_value = result['llm1_prediction']
                    imputed_df.iloc[row_idx, imputed_df.columns.get_loc(col_name)] = imputed_value
                    
                    # Store result
                    imputation_results.append({
                        'row_idx': row_idx,
                        'column': col_name,
                        'original_value': original_df.iloc[row_idx][col_name] if row_idx < len(original_df) else None,
                        'imputed_value': imputed_value,
                        'cluster_id': cluster_id,
                        'confidence': result['confidence'],
                        'reasoning': result['llm1_reasoning']
                    })
                    
                    print(f"  SUCCESS: {col_name} = {imputed_value}")
                else:
                    print(f"  FAILED: {col_name}")
                    imputation_results.append({
                        'row_idx': row_idx,
                        'column': col_name,
                        'original_value': original_df.iloc[row_idx][col_name] if row_idx < len(original_df) else None,
                        'imputed_value': None,
                        'cluster_id': cluster_id,
                        'confidence': 'Failed',
                        'reasoning': 'Imputation failed'
                    })
                
            except Exception as e:
                print(f"  ERROR: {col_name} - {e}")
                imputation_results.append({
                    'row_idx': row_idx,
                    'column': col_name,
                    'original_value': original_df.iloc[row_idx][col_name] if row_idx < len(original_df) else None,
                    'imputed_value': None,
                    'cluster_id': cluster_id,
                    'confidence': 'Error',
                    'reasoning': str(e)
                })
        
        return imputed_df, imputation_results
        
    except Exception as e:
        print(f"ERROR in imputation: {e}")
        return None, None

def evaluate_imputation(imputation_results, original_df):
    """
    Evaluate imputation results against original data
    """
    print("\nEVALUATING IMPUTATION RESULTS")
    print("=" * 60)
    
    # Separate successful and failed imputations
    successful_results = [r for r in imputation_results if r['imputed_value'] is not None]
    failed_results = [r for r in imputation_results if r['imputed_value'] is None]
    
    print(f"Successful imputations: {len(successful_results)}")
    print(f"Failed imputations: {len(failed_results)}")
    print(f"Success rate: {len(successful_results)/len(imputation_results)*100:.1f}%")
    
    if not successful_results:
        print("No successful imputations to evaluate")
        return {}
    
    # Calculate metrics for different data types
    numeric_columns = []
    categorical_columns = []
    
    for result in successful_results:
        col = result['column']
        if col in original_df.columns:
            if pd.api.types.is_numeric_dtype(original_df[col]):
                numeric_columns.append(result)
            else:
                categorical_columns.append(result)
    
    metrics = {}
    
    # Numeric columns evaluation
    if numeric_columns:
        print(f"\nNumeric columns evaluation ({len(numeric_columns)} cells):")
        
        # Convert to numeric for comparison
        numeric_original = []
        numeric_imputed = []
        
        for result in numeric_columns:
            try:
                orig_val = float(result['original_value']) if pd.notna(result['original_value']) else None
                imp_val = float(result['imputed_value']) if pd.notna(result['imputed_value']) else None
                
                if orig_val is not None and imp_val is not None:
                    numeric_original.append(orig_val)
                    numeric_imputed.append(imp_val)
            except:
                continue
        
        if numeric_original and numeric_imputed:
            mse = mean_squared_error(numeric_original, numeric_imputed)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(np.array(numeric_original) - np.array(numeric_imputed)))
            
            metrics['numeric'] = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'count': len(numeric_original)
            }
            
            print(f"  MSE: {mse:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAE: {mae:.4f}")
    
    # Categorical columns evaluation
    if categorical_columns:
        print(f"\nCategorical columns evaluation ({len(categorical_columns)} cells):")
        
        correct_predictions = 0
        total_predictions = 0
        
        for result in categorical_columns:
            if pd.notna(result['original_value']) and pd.notna(result['imputed_value']):
                total_predictions += 1
                if str(result['original_value']).lower() == str(result['imputed_value']).lower():
                    correct_predictions += 1
        
        if total_predictions > 0:
            accuracy = correct_predictions / total_predictions
            metrics['categorical'] = {
                'accuracy': accuracy,
                'correct': correct_predictions,
                'total': total_predictions
            }
            
            print(f"  Accuracy: {accuracy:.4f} ({correct_predictions}/{total_predictions})")
    
    return metrics

def save_results(imputed_df, imputation_results, metrics):
    """
    Save all results to files
    """
    print("\nSAVING RESULTS")
    print("=" * 60)
    
    # Create results directory
    os.makedirs("clustering_results/llm1_imputation", exist_ok=True)
    
    # Save imputed dataset
    imputed_file = "clustering_results/llm1_imputation/phone_imputed_llm1.csv"
    imputed_df.to_csv(imputed_file, index=False)
    print(f"Imputed dataset saved: {imputed_file}")
    
    # Save detailed results
    results_file = "clustering_results/llm1_imputation/imputation_results.json"
    with open(results_file, "w") as f:
        json.dump(imputation_results, f, indent=2)
    print(f"Detailed results saved: {results_file}")
    
    # Save metrics
    metrics_file = "clustering_results/llm1_imputation/evaluation_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Evaluation metrics saved: {metrics_file}")
    
    # Save summary
    summary = {
        'total_missing_cells': len(imputation_results),
        'successful_imputations': len([r for r in imputation_results if r['imputed_value'] is not None]),
        'failed_imputations': len([r for r in imputation_results if r['imputed_value'] is None]),
        'success_rate': len([r for r in imputation_results if r['imputed_value'] is not None]) / len(imputation_results) * 100,
        'metrics': metrics
    }
    
    summary_file = "clustering_results/llm1_imputation/summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved: {summary_file}")

def main():
    """
    Main function
    """
    print("LLM1 CLUSTERING-BASED IMPUTATION FOR PHONE DATASET")
    print("=" * 80)
    print("Using 10% missing phone dataset with LLM1 clustering imputation")
    print()
    
    # Load datasets
    original_df, missing_df, missing_cells = load_datasets()
    if original_df is None:
        return
    
    # Find missing cells
    missing_cell_list = find_missing_cells(missing_df)
    
    # Impute missing cells
    imputed_df, imputation_results = impute_missing_cells(missing_df, missing_cell_list, original_df)
    if imputed_df is None:
        return
    
    # Evaluate results
    metrics = evaluate_imputation(imputation_results, original_df)
    
    # Save results
    save_results(imputed_df, imputation_results, metrics)
    
    print(f"\n{'='*80}")
    print("LLM1 IMPUTATION COMPLETED")
    print(f"{'='*80}")
    print(f"Total missing cells: {len(imputation_results)}")
    print(f"Successful imputations: {len([r for r in imputation_results if r['imputed_value'] is not None])}")
    print(f"Success rate: {len([r for r in imputation_results if r['imputed_value'] is not None])/len(imputation_results)*100:.1f}%")
    
    if metrics:
        if 'numeric' in metrics:
            print(f"Numeric MSE: {metrics['numeric']['mse']:.4f}")
        if 'categorical' in metrics:
            print(f"Categorical Accuracy: {metrics['categorical']['accuracy']:.4f}")

if __name__ == "__main__":
    main()















