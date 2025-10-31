# Clustering Implementation Summary

## Date
**October 27, 2025**

## Objective
Apply clustering to all 4 datasets (phone, buy, zomato, restaurant) to enable cluster-based missing value imputation with at least 2 meaningful clusters per dataset.

## Clustering Methodology

### 1. **Approach: Hybrid DBSCAN + K-Means**
The script `fix_clusters_zomato_restaurant.py` uses a **two-phase clustering strategy**:

#### Phase 1: DBSCAN Clustering
- **Algorithm**: DBSCAN with Gower Distance
- **Purpose**: Identify natural groupings without pre-specifying number of clusters
- **Parameters Tested**: 
  - eps: [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0]
  - min_samples: [2, 3, 4, 5]
  - Total combinations: 48 per dataset

#### Phase 2: K-Means Clustering
- **Algorithm**: K-Means with Label Encoding + StandardScaler
- **Purpose**: Provide well-balanced clusters when DBSCAN produces too many tiny clusters
- **K Values Tested**: k = 2 to 10
- **Preprocessing**:
  1. Label encode categorical variables
  2. Standardize all features
  3. Apply K-Means clustering

### 2. **Algorithm Selection Logic**

```python
if dbscan_result and kmeans_result:
    if avg_cluster_size < 5:
        # Prefer K-Means for well-balanced clusters
        choose_kmeans()
    elif dbscan_score > kmeans_score:
        # Prefer DBSCAN if it provides better quality
        choose_dbscan()
    else:
        choose_kmeans()
```

### 3. **Quality Scoring System**

The script uses a customized scoring metric that rewards:
- **More clusters**: Higher score for more clusters
- **Fewer outliers**: Penalizes high noise ratio
- **Adequate cluster size**: Penalizes clusters with < 10 points (50% penalty)

```
Score = (num_clusters) × (1 - outlier_ratio) × size_penalty

where size_penalty = 0.5 if avg_cluster_size < 10, else 1.0
```

## Results Summary

### Dataset 1: PHONE ✅
- **Method Selected**: DBSCAN
- **Configuration**: eps=0.4, min_samples=2
- **Clusters**: 683 clusters
- **Outliers**: 2,102 (24.4%)
- **Score**: 258.30
- **Decision**: DBSCAN chosen (score > K-Means, adequate cluster sizes)
- **Largest Cluster**: 4,443 points (51.5% of data)

### Dataset 2: BUY ✅
- **Method Selected**: DBSCAN
- **Configuration**: eps=0.5, min_samples=2
- **Clusters**: 46 clusters
- **Outliers**: 327 (50.2%)
- **Score**: 11.45
- **Decision**: DBSCAN chosen (score > K-Means)
- **Largest Cluster**: 118 points (18.1% of data)

### Dataset 3: ZOMATO ✅
- **Method Selected**: K-Means (k=10)
- **Configuration**: eps=0.2, min_samples=2 (DBSCAN rejected)
- **Clusters**: 10 clusters
- **Outliers**: 0 (0%)
- **Score**: 0.126
- **Decision**: K-Means chosen (DBSCAN produced 728 tiny clusters with avg size 2.2)
- **Cluster Balance**: All clusters between 702-1,009 points (9.3%-11.9%)

### Dataset 4: RESTAURANT ✅
- **Method Selected**: DBSCAN
- **Configuration**: eps=0.6, min_samples=2
- **Clusters**: 65 clusters
- **Outliers**: 121 (14.0%)
- **Score**: 55.90
- **Decision**: DBSCAN chosen (score > K-Means)
- **Largest Cluster**: 123 points (14.2% of data)

## Storage of Results

### 1. **CSV Files with Cluster Labels**
**Location**: `clustering_results/[dataset]_with_clusters_[timestamp].csv`

#### Structure:
```csv
original_columns..., cluster_label, is_outlier
```

#### Files Generated:
1. `phone_with_clusters_20251027_202912.csv`
   - 8,628 rows × 42 columns (40 original + cluster_label + is_outlier)
   - 683 distinct clusters

2. `buy_with_clusters_20251027_202913.csv`
   - 651 rows × 6 columns (4 original + cluster_label + is_outlier)
   - 46 distinct clusters

3. `zomato_with_clusters_20251027_202913.csv`
   - 8,500 rows × 12 columns (10 original + cluster_label + is_outlier)
   - 10 distinct clusters

4. `restaurant_with_clusters_20251027_202913.csv`
   - 864 rows × 7 columns (5 original + cluster_label + is_outlier)
   - 65 distinct clusters

### 2. **JSON Metadata Files**
**Location**: `clustering_results/cluster_info_[dataset]_[timestamp].json`

#### Structure:
```json
{
  "dataset": "dataset_name",
  "method": "DBSCAN|KMeans",
  "n_clusters": number,
  "n_outliers": number,
  "score": float,
  "cluster_analysis": {
    "cluster_id": {
      "size": number_of_points,
      "characteristics": {
        "column_name": {
          "type": "categorical|numerical",
          "top_values": {...},  // for categorical
          "mean": float,        // for numerical
          "std": float,
          "min": float,
          "max": float
        }
      }
    }
  }
}
```

#### Files Generated:
1. `cluster_info_phone_20251027_202912.json` (4.5 MB)
   - Complete cluster analysis for 683 clusters
   - Characteristic statistics for each cluster

2. `cluster_info_buy_20251027_202913.json` (40 KB)
   - Complete cluster analysis for 46 clusters

3. `cluster_info_zomato_20251027_202913.json` (20 KB)
   - Complete cluster analysis for 10 clusters

4. `cluster_info_restaurant_20251027_202913.json` (61 KB)
   - Complete cluster analysis for 65 clusters

### 3. **Usage in 3-LLM Pipeline**

#### For LLM1 (Cluster-based Imputation):
```python
# Load cluster information
import json
with open('clustering_results/cluster_info_phone.json', 'r') as f:
    cluster_info = json.load(f)

# Get cluster characteristics
cluster_id = 0
characteristics = cluster_info['cluster_analysis'][str(cluster_id)]

# For each column in the cluster:
for col_name, stats in characteristics['characteristics'].items():
    if stats['type'] == 'categorical':
        # Use top value (mode)
        imputation_value = list(stats['top_values'].keys())[0]
    else:
        # Use mean value
        imputation_value = stats['mean']
```

#### For Assigning Test Data to Clusters:
```python
# Load clustered training data
df_clustered = pd.read_csv('clustering_results/phone_with_clusters.csv')

# For each test row, find nearest cluster based on Gower distance
# or use K-Means prediction for assigned cluster
```

## Why Hybrid Approach Works

### DBSCAN Advantages:
✅ Handles natural groupings without pre-specifying k
✅ Identifies outliers naturally
✅ Works with arbitrary cluster shapes
✅ Good for datasets with natural clusters (Phone, Buy, Restaurant)

### DBSCAN Disadvantages:
❌ Can produce too many tiny clusters (Zomato: 728 clusters)
❌ Parameter-sensitive
❌ High noise ratio in some cases

### K-Means Advantages:
✅ Produces well-balanced clusters
✅ No outliers
✅ Fast and efficient
✅ Good for evenly distributed data (Zomato)

### K-Means Disadvantages:
❌ Requires pre-specifying k
❌ Assumes spherical clusters
❌ Sensitive to initialization

### Hybrid Solution:
**Best of Both Worlds**: Automatically selects the method that produces the best clusters for each specific dataset.

## Technical Implementation Details

### 1. **Gower Distance for Mixed Data**
```python
import gower

# Calculate distance matrix
dist_matrix = gower.gower_matrix(df_processed)
# Handles both categorical and numerical columns automatically
```

### 2. **Preprocessing**
```python
def preprocess_for_clustering(df):
    # Fill missing categorical with 'Unknown'
    # Fill missing numerical with median
    # Ensure proper data types
    return df_processed
```

### 3. **Quality Assessment**
```python
# Calculate score considering:
# - Number of clusters (more is better)
# - Outlier ratio (less is better)
# - Average cluster size (adequate is better)

score = n_clusters × (1 - outlier_ratio) × size_penalty
```

## Summary Statistics

| Dataset | Rows | Clusters | Method | Outliers % | File Size (JSON) |
|---------|------|----------|--------|-----------|------------------|
| Phone | 8,628 | 683 | DBSCAN | 24.4% | 4.5 MB |
| Buy | 651 | 46 | DBSCAN Conserva% | 40 KB |
| Zomato | 8,500 | 10 | K-Means | 0% | 20 KB |
| Restaurant | 864 | 65 | DBSCAN | 14.0% | 61 KB |
| **Total** | **18,643** | **804** | **Hybrid** | **13.8%** | **4.6 MB** |

## Impact on Imputation Pipeline

### Before (Old Clusters):
- Phone: 4 clusters (limited)
- Buy: 9 clusters (limited)
- Zomato: 1 cluster (useless)
- Restaurant: 0 clusters (failed)
- **Total usable clusters: 13**

### After (New Clusters):
- Phone: 683 clusters (excellent granularity)
- Buy: 46 clusters (good granularity)
- Zomato: 10 clusters (well-balanced)
- Restaurant: 65 clusters (good separation)
- **Total usable clusters: 804**

### Improvement:
✅ **62x increase in total clusters**
✅ All datasets have meaningful clusters
✅ Better cluster-specific context for LLM1
✅ Expected improvement in imputation accuracy

## Next Steps

1. **Update LLM1 Pipeline** to load new cluster information
2. **Test Imputation Accuracy** with new clusters
3. **Compare Results** before vs after cluster fix
4. **Optimize Cluster Usage** for fastest runtime

---
*Generated: October 27, 2025*
*Script: fix_clusters_zomato_restaurant.py*
*Total Runtime: ~15 minutes for all 4 datasets*





