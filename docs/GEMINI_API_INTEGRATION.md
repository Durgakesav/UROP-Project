# 🤖 Gemini API Integration for 3LLM Pipeline

## Overview

This guide shows how to integrate Google's Gemini API for LLM1 (cluster-based imputation) in your 3LLM pipeline.

## 🚀 Quick Start

### 1. Get Your Gemini API Key

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy your API key

### 2. Install Dependencies

```bash
pip install -r requirements_gemini.txt
```

### 3. Set Up API Key

**Option A: Environment Variable (Recommended)**
```bash
# Windows
set GEMINI_API_KEY=your_api_key_here

# Linux/Mac
export GEMINI_API_KEY=your_api_key_here
```

**Option B: Direct in Script**
```python
GEMINI_API_KEY = "your_actual_api_key_here"
```

### 4. Run the Pipeline

```bash
python run_gemini_llm1.py
```

---

## 📁 File Structure

```
scripts/
├── gemini_llm1_pipeline.py    # Main Gemini integration
├── implement_3llm_pipeline.py # Original pipeline
└── implement_gemini_llm1.py   # Previous Gemini attempt

clustering_results/
├── cluster_info_phone.json    # Phone cluster centroids
├── cluster_info_buy.json      # Buy cluster centroids
├── cluster_info_restaurant.json # Restaurant analysis
└── zomato_cluster_analysis.json # Zomato characteristics

run_gemini_llm1.py            # Usage script
requirements_gemini.txt       # Dependencies
```

---

## 🔧 Implementation Details

### **GeminiLLM1Pipeline Class**

The main class that integrates Gemini API for LLM1:

```python
class GeminiLLM1Pipeline:
    def __init__(self, gemini_api_key):
        # Configure Gemini API
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-pro')
    
    def step2_llm1_gemini_imputation(self, missing_row, target_column, cluster_id):
        # Uses Gemini API for cluster-based imputation
        response = self.model.generate_content(prompt)
        return prediction, prompt, reasoning
```

### **Key Features**

1. **✅ Cluster Integration**: Uses pre-computed cluster centroids
2. **✅ Structured Prompts**: Well-formatted prompts for better results
3. **✅ Response Parsing**: Extracts prediction and reasoning
4. **✅ Error Handling**: Robust error handling for API calls
5. **✅ Fallback Logic**: Handles empty or malformed responses

---

## 📊 Usage Examples

### **Basic Usage**

```python
from scripts.gemini_llm1_pipeline import GeminiLLM1Pipeline

# Initialize with API key
pipeline = GeminiLLM1Pipeline("your_api_key")

# Run pipeline
results = pipeline.run_gemini_pipeline(
    train_file="train_sets/phone_train_original.csv",
    test_file="test_sets/phone_test_MNAR.csv",
    dataset_name="phone",
    missing_row_idx=0,
    target_column="brand"
)
```

### **Advanced Usage**

```python
# Load cluster info separately
pipeline.load_cluster_info("phone")

# Run individual steps
prediction, prompt, reasoning = pipeline.step2_llm1_gemini_imputation(
    missing_row, "brand", "0"
)
```

---

## 🎯 Prompt Engineering

### **LLM1 Prompt Structure**

The Gemini API receives structured prompts:

```
You are an expert data imputation specialist with access to cluster-specific data.

CONTEXT:
- You have access to data from cluster {cluster_id}
- Cluster centroid: {centroid_data}

TASK:
Predict the missing value for column '{target_column}' in this row:
{missing_row_data}

ANALYSIS REQUIREMENTS:
1. Analyze the cluster data and centroid
2. Consider the missing row's existing values
3. Predict the most likely value
4. Provide reasoning

RESPONSE FORMAT:
PREDICTION: [value]
REASONING: [explanation]
```

### **Response Parsing**

The system automatically parses responses:

```python
# Extract structured response
if "PREDICTION:" in response_text:
    prediction = extract_prediction(response_text)

if "REASONING:" in response_text:
    reasoning = extract_reasoning(response_text)
```

---

## 📈 Results Format

### **Output Structure**

```json
{
  "missing_row_idx": 0,
  "target_column": "brand",
  "cluster_id": "0",
  "llm1_prediction": "Samsung",
  "llm1_reasoning": "Based on cluster centroid showing Samsung dominance",
  "llm2_prediction": "Samsung",
  "final_prediction": "Samsung",
  "confidence": "High (Agreement)",
  "gemini_api_used": true
}
```

### **Key Metrics**

- **Prediction Accuracy**: Compare with ground truth
- **Reasoning Quality**: Assess explanation clarity
- **Response Time**: Monitor API latency
- **Error Rate**: Track failed API calls

---

## 🔍 Testing & Validation

### **Test Cases**

1. **Phone Dataset**: Test brand prediction
2. **Buy Dataset**: Test manufacturer prediction
3. **Zomato Dataset**: Test cuisine prediction
4. **Restaurant Dataset**: Test type prediction

### **Validation Steps**

```python
# Test API connectivity
pipeline = GeminiLLM1Pipeline(api_key)
test_response = pipeline.model.generate_content("Test message")

# Test cluster loading
success = pipeline.load_cluster_info("phone")

# Test full pipeline
results = pipeline.run_gemini_pipeline(...)
```

---

## 🛠️ Troubleshooting

### **Common Issues**

1. **API Key Invalid**
   ```
   Error: 403 Forbidden
   Solution: Check API key validity
   ```

2. **Rate Limiting**
   ```
   Error: 429 Too Many Requests
   Solution: Add delays between requests
   ```

3. **Empty Response**
   ```
   Error: Empty response from Gemini API
   Solution: Check prompt format and retry
   ```

4. **Cluster File Missing**
   ```
   Error: Cluster file not found
   Solution: Run clustering first
   ```

### **Debug Mode**

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run pipeline with debug info
results = pipeline.run_gemini_pipeline(...)
```

---

## 🚀 Next Steps

### **Immediate Improvements**

1. **Add LLM2 Gemini Integration**: Extend to RAG-based imputation
2. **Add LLM3 Gemini Integration**: Implement comparison logic
3. **Batch Processing**: Handle multiple missing values
4. **Caching**: Cache API responses for efficiency

### **Advanced Features**

1. **Multi-Model Support**: Support different Gemini models
2. **Custom Prompts**: Allow user-defined prompt templates
3. **Confidence Scoring**: Implement sophisticated confidence metrics
4. **A/B Testing**: Compare different prompt strategies

---

## 📚 Resources

- [Gemini API Documentation](https://ai.google.dev/docs)
- [Google AI Studio](https://makersuite.google.com/)
- [3LLM Pipeline Guide](docs/3LLM_PIPELINE_GUIDE.md)
- [Clustering Results](clustering_results/)

---

**Status**: ✅ **Gemini API Integration Complete for LLM1**  
**Next Phase**: Extend to LLM2 and LLM3 for complete pipeline
