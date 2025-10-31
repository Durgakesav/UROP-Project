"""
Test script for Gemini API integration
This script tests the Gemini API connection without requiring a full pipeline run
"""

import os
import google.generativeai as genai

def test_gemini_api(api_key):
    """
    Test basic Gemini API connectivity
    """
    try:
        # Configure API
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        
        # Simple test
        print("🧪 Testing Gemini API connection...")
        response = model.generate_content("Hello, can you respond with 'API test successful'?")
        
        if response.text:
            print(f"✅ API Response: {response.text.strip()}")
            return True
        else:
            print("❌ Empty response from API")
            return False
            
    except Exception as e:
        print(f"❌ API Error: {e}")
        return False

def test_cluster_loading():
    """
    Test loading cluster information
    """
    try:
        import json
        
        cluster_files = [
            "clustering_results/cluster_info_phone.json",
            "clustering_results/cluster_info_buy.json",
            "clustering_results/cluster_info_restaurant.json"
        ]
        
        print("\n📁 Testing cluster file loading...")
        
        for file_path in cluster_files:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    data = json.load(f)
                print(f"✅ {file_path}: {len(data.get('clusters', {}))} clusters")
            else:
                print(f"❌ {file_path}: File not found")
        
        return True
        
    except Exception as e:
        print(f"❌ Cluster loading error: {e}")
        return False

def main():
    """
    Main test function
    """
    print("🚀 Gemini API Integration Test")
    print("=" * 50)
    
    # Test 1: API Key
    api_key = os.getenv('GEMINI_API_KEY')
    
    if not api_key:
        print("❌ No API key found in environment variables")
        print("\nTo set your API key:")
        print("1. Get your key from: https://makersuite.google.com/app/apikey")
        print("2. Set it as environment variable:")
        print("   - Windows: set GEMINI_API_KEY=your_key_here")
        print("   - Linux/Mac: export GEMINI_API_KEY=your_key_here")
        
        # Ask for manual input
        manual_key = input("\nOr enter your API key manually (press Enter to skip): ").strip()
        if manual_key:
            api_key = manual_key
        else:
            print("Skipping API test...")
            api_key = None
    
    # Test 2: API Connection
    if api_key:
        api_success = test_gemini_api(api_key)
    else:
        api_success = False
    
    # Test 3: Cluster Files
    cluster_success = test_cluster_loading()
    
    # Test 4: Data Files
    print("\n📊 Testing data file availability...")
    data_files = [
        "train_sets/phone_train_original.csv",
        "test_sets/phone_test_MNAR.csv"
    ]
    
    data_success = True
    for file_path in data_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}: Available")
        else:
            print(f"❌ {file_path}: Not found")
            data_success = False
    
    # Summary
    print(f"\n{'='*50}")
    print("📋 TEST SUMMARY")
    print(f"{'='*50}")
    print(f"API Connection: {'✅ Success' if api_success else '❌ Failed'}")
    print(f"Cluster Files: {'✅ Success' if cluster_success else '❌ Failed'}")
    print(f"Data Files: {'✅ Success' if data_success else '❌ Failed'}")
    
    if api_success and cluster_success and data_success:
        print(f"\n🎉 All tests passed! Ready to run Gemini LLM1 pipeline.")
        print(f"\nNext steps:")
        print(f"1. Run: python run_gemini_llm1.py")
        print(f"2. Or use: python scripts/gemini_llm1_pipeline.py")
    else:
        print(f"\n⚠️  Some tests failed. Please fix the issues above.")
        
        if not api_success:
            print(f"\nAPI Issues:")
            print(f"- Check your API key is valid")
            print(f"- Ensure you have internet connection")
            print(f"- Verify API quotas are not exceeded")
        
        if not cluster_success:
            print(f"\nCluster Issues:")
            print(f"- Run clustering analysis first")
            print(f"- Check clustering_results/ directory exists")
        
        if not data_success:
            print(f"\nData Issues:")
            print(f"- Ensure train_sets/ and test_sets/ directories exist")
            print(f"- Run data preparation scripts first")

if __name__ == "__main__":
    main()
