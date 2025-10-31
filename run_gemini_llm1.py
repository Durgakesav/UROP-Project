"""
Gemini API Integration for 3LLM Pipeline - Usage Example
This script demonstrates how to use the Gemini API for LLM1 in the 3LLM pipeline
"""

import os
import json
from scripts.gemini_llm1_pipeline import GeminiLLM1Pipeline

def main():
    """
    Main function to run the Gemini LLM1 pipeline
    """
    print("🚀 Gemini API Integration for 3LLM Pipeline")
    print("=" * 60)
    
    # Get Gemini API key
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    
    if not gemini_api_key:
        print("❌ Gemini API key not found!")
        print("\nTo use this script:")
        print("1. Get your API key from: https://makersuite.google.com/app/apikey")
        print("2. Set it as an environment variable:")
        print("   - Windows: set GEMINI_API_KEY=your_api_key_here")
        print("   - Linux/Mac: export GEMINI_API_KEY=your_api_key_here")
        print("3. Or modify the script to include your API key directly")
        return
    
    try:
        # Initialize pipeline
        pipeline = GeminiLLM1Pipeline(gemini_api_key)
        
        # Test on phone dataset
        print("\n📱 Testing on Phone Dataset...")
        results = pipeline.run_gemini_pipeline(
            train_file="train_sets/phone_train_original.csv",
            test_file="test_sets/phone_test_MNAR.csv",
            dataset_name="phone",
            missing_row_idx=0,  # First row with missing values
            target_column="brand"  # Column to impute
        )
        
        if results:
            # Save results
            output_file = "clustering_results/gemini_llm1_results.json"
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)
            
            print(f"\n✅ Results saved to: {output_file}")
            
            # Display summary
            print(f"\n📊 SUMMARY:")
            print(f"   Target Column: {results['target_column']}")
            print(f"   LLM1 (Gemini) Prediction: {results['llm1_prediction']}")
            print(f"   LLM1 Reasoning: {results['llm1_reasoning']}")
            print(f"   Final Prediction: {results['final_prediction']}")
            print(f"   Confidence: {results['confidence']}")
            
        else:
            print("❌ Pipeline failed")
            
    except Exception as e:
        print(f"❌ Error running pipeline: {e}")
        print("\nTroubleshooting:")
        print("1. Check your Gemini API key is valid")
        print("2. Ensure you have internet connection")
        print("3. Verify the cluster files exist in clustering_results/")
        print("4. Check that train/test files exist")

def test_with_custom_api_key():
    """
    Test function with direct API key input
    """
    print("🔑 Enter your Gemini API key:")
    api_key = input("API Key: ").strip()
    
    if not api_key:
        print("❌ No API key provided")
        return
    
    try:
        pipeline = GeminiLLM1Pipeline(api_key)
        
        # Quick test
        print("\n🧪 Quick Test...")
        results = pipeline.run_gemini_pipeline(
            train_file="train_sets/phone_train_original.csv",
            test_file="test_sets/phone_test_MNAR.csv",
            dataset_name="phone",
            missing_row_idx=0,
            target_column="brand"
        )
        
        if results:
            print("✅ Test successful!")
            print(f"   Prediction: {results['llm1_prediction']}")
            print(f"   Reasoning: {results['llm1_reasoning']}")
        else:
            print("❌ Test failed")
            
    except Exception as e:
        print(f"❌ Test error: {e}")

if __name__ == "__main__":
    print("Choose an option:")
    print("1. Use environment variable for API key")
    print("2. Enter API key manually")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        main()
    elif choice == "2":
        test_with_custom_api_key()
    else:
        print("❌ Invalid choice")
