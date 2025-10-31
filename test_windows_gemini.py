"""
Windows-specific Gemini API test
This script tests the Gemini API setup on Windows
"""

import os
import sys

def test_windows_setup():
    """
    Test Gemini API setup on Windows
    """
    print("Windows Gemini API Setup Test")
    print("=" * 40)
    
    # Test 1: Python version
    print(f"Python version: {sys.version}")
    
    # Test 2: Check if Gemini package is installed
    try:
        import google.generativeai as genai
        print("Gemini package: Installed")
    except ImportError:
        print("Gemini package: NOT INSTALLED")
        print("Run: pip install google-generativeai")
        return False
    
    # Test 3: Check environment variable
    api_key = os.getenv('GEMINI_API_KEY')
    if api_key:
        print("API key: Found in environment")
        print(f"Key length: {len(api_key)} characters")
    else:
        print("API key: NOT FOUND in environment")
        print("\nTo set API key on Windows:")
        print("Command Prompt: set GEMINI_API_KEY=your_key_here")
        print("PowerShell: $env:GEMINI_API_KEY=\"your_key_here\"")
    
    # Test 4: Check cluster files
    cluster_files = [
        "clustering_results\\cluster_info_phone.json",
        "clustering_results\\cluster_info_buy.json"
    ]
    
    print("\nCluster files:")
    for file_path in cluster_files:
        if os.path.exists(file_path):
            print(f"  {file_path}: Found")
        else:
            print(f"  {file_path}: NOT FOUND")
    
    # Test 5: Check data files
    data_files = [
        "train_sets\\phone_train_original.csv",
        "test_sets\\phone_test_MNAR.csv"
    ]
    
    print("\nData files:")
    for file_path in data_files:
        if os.path.exists(file_path):
            print(f"  {file_path}: Found")
        else:
            print(f"  {file_path}: NOT FOUND")
    
    return True

def test_gemini_connection():
    """
    Test actual Gemini API connection
    """
    api_key = os.getenv('GEMINI_API_KEY')
    
    if not api_key:
        print("\nNo API key found. Cannot test connection.")
        return False
    
    try:
        import google.generativeai as genai
        
        print("\nTesting Gemini API connection...")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-pro')
        
        response = model.generate_content("Say 'Windows test successful'")
        
        if response.text:
            print(f"API Response: {response.text.strip()}")
            print("Connection test: SUCCESS")
            return True
        else:
            print("Connection test: FAILED (empty response)")
            return False
            
    except Exception as e:
        print(f"Connection test: FAILED ({e})")
        return False

def main():
    """
    Main test function
    """
    print("Windows Gemini API Test")
    print("=" * 30)
    
    # Run setup test
    setup_ok = test_windows_setup()
    
    if setup_ok:
        # Run connection test
        connection_ok = test_gemini_connection()
        
        if connection_ok:
            print("\nAll tests passed! Ready to use Gemini API.")
            print("\nNext steps:")
            print("1. Run: python demo_gemini_api.py")
            print("2. Or run: python run_gemini_llm1.py")
        else:
            print("\nSetup OK but connection failed.")
            print("Check your API key and internet connection.")
    else:
        print("\nSetup issues found. Please fix them first.")

if __name__ == "__main__":
    main()
