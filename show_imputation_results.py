"""
Show Before and After Imputation with Gemini LLM1
This script shows the complete column before and after imputation
"""

import pandas as pd
import json
import os
from scripts.gemini_llm1_pipeline import GeminiLLM1Pipeline

# Your Gemini API key
GEMINI_API_KEY = "AIzaSyC6yMHoOO60kywdWgBWcwUtB2nbTo8_kAg"

def show_before_after_imputation():
    """
    Show complete column before and after LLM1 imputation
    """
    print("BEFORE AND AFTER IMPUTATION WITH GEMINI LLM1")
    print("=" * 70)
    
    try:
        # Initialize pipeline
        pipeline = GeminiLLM1Pipeline(GEMINI_API_KEY)
        
        # Load test data
        test_file = "test_sets/phone_test_MNAR.csv"
        train_file = "train_sets/phone_train_original.csv"
        
        if not os.path.exists(test_file) or not os.path.exists(train_file):
            print(f"ERROR: Required files not found")
            print(f"Test file: {test_file} - {'EXISTS' if os.path.exists(test_file) else 'NOT FOUND'}")
            print(f"Train file: {train_file} - {'EXISTS' if os.path.exists(train_file) else 'NOT FOUND'}")
            return
        
        # Load datasets
        test_df = pd.read_csv(test_file)
        train_df = pd.read_csv(train_file)
        
        print(f"\nDATASET INFO:")
        print(f"Training data: {len(train_df)} rows, {len(train_df.columns)} columns")
        print(f"Test data: {len(test_df)} rows, {len(test_df.columns)} columns")
        
        # Show target column before imputation
        target_column = "brand"
        print(f"\n{'='*70}")
        print(f"BEFORE IMPUTATION - Column '{target_column}'")
        print(f"{'='*70}")
        
        # Show first 20 rows of the target column
        print(f"First 20 values in '{target_column}' column:")
        print("-" * 50)
        for i in range(min(20, len(test_df))):
            value = test_df.iloc[i][target_column]
            status = "MISSING" if pd.isna(value) else "PRESENT"
            print(f"Row {i:2d}: {str(value):15s} [{status}]")
        
        # Count missing values
        missing_count = test_df[target_column].isna().sum()
        total_count = len(test_df)
        print(f"\nMissing values: {missing_count}/{total_count} ({missing_count/total_count*100:.1f}%)")
        
        # Run LLM1 imputation on first missing row
        print(f"\n{'='*70}")
        print(f"RUNNING GEMINI LLM1 IMPUTATION")
        print(f"{'='*70}")
        
        # Find first missing row
        missing_rows = test_df[test_df[target_column].isna()]
        if len(missing_rows) == 0:
            print("No missing values found in target column")
            return
        
        missing_row_idx = missing_rows.index[0]
        missing_row = test_df.iloc[missing_row_idx]
        
        print(f"Processing missing row {missing_row_idx}:")
        print(f"Target column '{target_column}': MISSING")
        print(f"Other columns:")
        for col in test_df.columns:
            if col != target_column:
                value = missing_row[col]
                if not pd.isna(value):
                    print(f"  {col}: {value}")
        
        # Run Gemini LLM1 imputation
        print(f"\nRunning Gemini LLM1 pipeline...")
        results = pipeline.run_gemini_pipeline(
            train_file=train_file,
            test_file=test_file,
            dataset_name="phone",
            missing_row_idx=missing_row_idx,
            target_column=target_column
        )
        
        if results:
            print(f"\n{'='*70}")
            print(f"AFTER IMPUTATION - Column '{target_column}'")
            print(f"{'='*70}")
            
            # Show the imputed value
            imputed_value = results['llm1_prediction']
            reasoning = results['llm1_reasoning']
            
            print(f"Row {missing_row_idx} IMPUTED VALUE:")
            print(f"  Before: MISSING")
            print(f"  After:  {imputed_value}")
            print(f"  Reasoning: {reasoning}")
            
            # Show updated column (first 20 rows)
            print(f"\nUpdated column '{target_column}' (first 20 rows):")
            print("-" * 50)
            
            # Create a copy of the test data and update the missing value
            test_df_updated = test_df.copy()
            test_df_updated.iloc[missing_row_idx, test_df_updated.columns.get_loc(target_column)] = imputed_value
            
            for i in range(min(20, len(test_df_updated))):
                value = test_df_updated.iloc[i][target_column]
                status = "MISSING" if pd.isna(value) else "PRESENT"
                if i == missing_row_idx:
                    status = "IMPUTED"
                print(f"Row {i:2d}: {str(value):15s} [{status}]")
            
            # Show statistics
            print(f"\nIMPUTATION STATISTICS:")
            print(f"  Original missing: {missing_count}/{total_count}")
            print(f"  After imputation: {test_df_updated[target_column].isna().sum()}/{total_count}")
            print(f"  Imputed value: {imputed_value}")
            print(f"  Confidence: {results['confidence']}")
            
            # Save updated dataset
            output_file = "clustering_results/phone_test_with_imputation.csv"
            test_df_updated.to_csv(output_file, index=False)
            print(f"\nUpdated dataset saved to: {output_file}")
            
        else:
            print("ERROR: Imputation failed")
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    show_before_after_imputation()















