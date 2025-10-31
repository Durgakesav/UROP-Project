"""
Simple Gemini API Demo for 3LLM Pipeline
This script demonstrates how to use Gemini API for LLM1 without emojis
"""

import os
import json
import google.generativeai as genai

def demo_gemini_api():
    """
    Demonstrate Gemini API usage for data imputation
    """
    print("Gemini API Demo for 3LLM Pipeline")
    print("=" * 50)
    
    # Get API key
    api_key = os.getenv('GEMINI_API_KEY')
    
    if not api_key:
        print("No API key found in environment variables")
        print("To get your API key:")
        print("1. Visit: https://makersuite.google.com/app/apikey")
        print("2. Create an API key")
        print("3. Set it as environment variable:")
        print("   Windows: set GEMINI_API_KEY=your_key_here")
        print("   Linux/Mac: export GEMINI_API_KEY=your_key_here")
        
        manual_key = input("\nEnter your API key manually (or press Enter to skip): ").strip()
        if manual_key:
            api_key = manual_key
        else:
            print("Skipping demo...")
            return
    
    try:
        # Configure Gemini
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-pro')
        
        print("\nTesting Gemini API connection...")
        
        # Test basic connection
        response = model.generate_content("Say 'API test successful'")
        print(f"API Response: {response.text.strip()}")
        
        # Load cluster data
        print("\nLoading cluster information...")
        cluster_file = "clustering_results/cluster_info_phone.json"
        
        if os.path.exists(cluster_file):
            with open(cluster_file, 'r') as f:
                cluster_data = json.load(f)
            
            print(f"Loaded {len(cluster_data['clusters'])} clusters for phone dataset")
            
            # Demo cluster-based prediction
            print("\nDemo: Cluster-based prediction")
            
            # Sample missing row (simulated)
            missing_row = {
                'brand': 'Celkon',
                'model': 'A63',
                'network_technology': 'GSM',
                'approx_price_EUR': 50.0
            }
            
            # Get first cluster centroid
            cluster_id = list(cluster_data['clusters'].keys())[0]
            centroid = cluster_data['clusters'][cluster_id]
            
            # Create prompt
            prompt = f"""
You are an expert data imputation specialist.

CONTEXT:
- You have access to data from cluster {cluster_id}
- Cluster centroid: {json.dumps(centroid, indent=2)}

TASK:
Predict the missing value for column 'brand' in this row:
{json.dumps(missing_row, indent=2)}

Based on the cluster data, what should be the value for 'brand'?
Provide only the predicted value.
"""
            
            print("Sending request to Gemini...")
            response = model.generate_content(prompt)
            
            if response.text:
                print(f"Gemini Prediction: {response.text.strip()}")
                print("\nDemo completed successfully!")
            else:
                print("Empty response from Gemini")
                
        else:
            print(f"Cluster file not found: {cluster_file}")
            
    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check your API key is valid")
        print("2. Ensure you have internet connection")
        print("3. Verify API quotas are not exceeded")

if __name__ == "__main__":
    demo_gemini_api()
