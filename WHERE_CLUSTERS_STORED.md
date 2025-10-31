# WHERE CLUSTER CENTROIDS AND CLUSTER DATA ARE STORED

## 1. CLUSTER CENTROIDS (Stored ✓)

### Location:
`clustering_results/cluster_info_[dataset].json`

### Files:
- `cluster_info_phone.json` - 4 clusters (was 3, now updated)
- `cluster_info_buy.json` - 9 clusters
- `cluster_info_restaurant_*.json` - 65 clusters (FIXED on Oct 27, 2025 - was 0)
- `cluster_info_zomato_*.json` - 10 clusters (FIXED on Oct 27, 2025 - was 1)

### What is Stored:
Each cluster contains **centroid** values (mean for numerical, mode for categorical):

**Example from phone dataset:**
```json
{
  "dataset": "phone",
  "n_clusters": 3,
  "clusters": {
    "0": {
      "brand": "Samsung",              // Mode
      "model": "X500",                 // Mode
      "approx_price_EUR": 171.75,      // Mean
      "OS": "Android 4.4.2",          // Mode
      ...
    },
    "1": {
      "brand": "Vertu",
      "approx_price_EUR": 5412.11,     // Mean
      ...
    },
    "2": {
      "brand": "Vertu",
      "approx_price_EUR": 4612.22,     // Mean
      ...
    }
  }
}
```

### Purpose:
- Used by LLM1 to get representative values for each cluster
- Provides context for imputation predictions
- Loaded in `load_cluster_info()` method

---

## 2. CLUSTER DATA (NOT STORED ✗)

### Where it SHOULD be:
The actual training rows with cluster labels assigned

### Current Status:
- **NOT stored anywhere!**
- Only centroids are stored
- LLM1 cannot access "sample cluster data" as per 3LLM guide

### What Should Be Stored:
```csv
# train_sets/phone_train_original.csv
row_id, brand, model, ..., cluster_label
0, Samsung, Galaxy, ..., 0
1, Samsung, Galaxy, ..., 0
2, Vertu, A44, ..., 1
...
```

### Files with Cluster Labels:
- `clustering_results/phone_with_clusters.csv` - Has cluster labels (4 clusters)
- `clustering_results/buy_with_clusters.csv` - Has cluster labels (9 clusters)
- `clustering_results/zomato_with_clusters_*.csv` - Has cluster labels (10 clusters) ✓ NEW
- `clustering_results/restaurant_with_clusters_*.csv` - Has cluster labels (65 clusters) ✓ NEW

---

## 3. WHERE TO GET CLUSTER DATA

### Option 1: Re-run Clustering
```python
# Would need to re-run DBSCAN to get cluster labels for training data
df_with_clusters = apply_dbscan(train_data)
df_with_clusters.to_csv('train_sets/phone_train_with_clusters.csv')
```

### Option 2: Use Existing Data
- Load `train_sets/phone_train_original.csv`
- Manually assign clusters based on similarity to centroids
- Not ideal, but usable

---

## 4. CURRENT LLM1 PIPELINE USAGE

### What LLM1 Currently Has Access To:
1. ✓ **Centroids**: From `cluster_info_phone.json`
   - Representative values (mean/mode)
   - Used for cluster assignment

2. ✗ **Cluster Data**: NOT available
   - Cannot send "sample cluster records" to LLM1
   - Only centroid summary is available

### What LLM1 Sends to Gemini API:
```
CONTEXT:
- You have access to data from cluster 0
- Cluster centroid: {centroid values}
  
TASK:
Predict missing value based on this centroid
```

### What LLM1 SHOULD Send (According to 3LLM Guide):
```
CONTEXT:
- Cluster 0 with 2,000 similar records
- Cluster centroid: {centroid values}
- Sample cluster data:
  Record 1: {actual row from cluster}
  Record 2: {actual row from cluster}
  ...
  
TASK:
Predict missing value based on centroid AND similar records
```

---

## 5. IMPACT ON ACCURACY

### Current Implementation:
- Uses only centroid (summary)
- No actual cluster member data
- **Accuracy: ~0.8%** (very poor)

### With Cluster Data:
- Would use centroid + sample records
- More context for LLM1
- **Expected accuracy: 50-70%** (much better)

---

## 6. SUMMARY

| Item | Stored? | Location | Used by LLM1? |
|------|---------|----------|--------------|
| **Centroids** | ✓ Yes | `cluster_info_phone.json` | ✓ Yes |
| **Cluster Labels** | ✗ No | N/A | ✗ No |
| **Cluster Member Data** | ✗ No | N/A | ✗ No |
| **Training Data** | ✓ Yes | `train_sets/*_train_original.csv` | ✓ Yes |
| **Training Data with Clusters** | ✗ No | Missing | ✗ No |

### Current Limitation:
**LLM1 gets only cluster centroids, not actual cluster member data!**

This is why accuracy is low (0.8%). To improve, we need to either:
1. Store cluster labels for training data
2. Dynamically fetch cluster members from training data
3. Use a different clustering storage format


