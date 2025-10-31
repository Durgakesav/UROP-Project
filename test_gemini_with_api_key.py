"""
Test Gemini API with your API key
This script tests the full Gemini LLM1 pipeline
"""

import json
import os
from scripts.gemini_llm1_pipeline import GeminiLLM1Pipeline

# Your Gemini API key
GEMINI_API_KEY = "AIzaSyC6yMHoOO60kywdWgBWcwUtB2nbTo8_kAg"

def test_full_pipeline():
    """
    Test the complete Gemini LLM1 pipeline
    """
    print("Testing Gemini LLM1 Pipeline with your API key")
    print("=" * 60)
    
    try:
        # Initialize pipeline with your API key
        pipeline = GeminiLLM1Pipeline(GEMINI_API_KEY)
        
        # Test on phone dataset
        print("\nTesting on Phone Dataset...")
        results = pipeline.run_gemini_pipeline(
            train_file="train_sets/phone_train_original.csv",
            test_file="test_sets/phone_test_MNAR.csv",
            dataset_name="phone",
            missing_row_idx=0,
            target_column="brand"
        )
        
        if results:
            # Save results
            output_file = "clustering_results/gemini_llm1_results.json"
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)
            
            print(f"\nResults saved to: {output_file}")
            print("\nSUMMARY:")
            print(f"  Target Column: {results['target_column']}")
            print(f"  LLM1 (Gemini) Prediction: {results['llm1_prediction']}")
            print(f"  LLM1 Reasoning: {results['llm1_reasoning']}")
            print(f"  Final Prediction: {results['final_prediction']}")
            print(f"  Confidence: {results['confidence']}")
            
            print("\nSUCCESS: Pipeline completed successfully!")
            return True
        else:
            print("Pipeline failed")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_full_pipeline()
