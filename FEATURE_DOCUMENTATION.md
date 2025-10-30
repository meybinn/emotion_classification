# Feature Engineering & Model Input Documentation

## 📊 Question 1: What Features Are Used for Training?

### ✅ **Statistical Features (Aggregated from Raw HR)**

The project uses **pre-computed statistical features** extracted from raw heart rate time series data. No raw arrays are used directly.

#### **Primary Features (Used in Production API):**

```python
features = [
    "mean_hr",      # Mean/average heart rate
    "std_hr",       # Standard deviation of heart rate
    "range_hr"      # Range (max - min) of heart rate
]
```

**Shape:** `(n_samples, 3)` - Fixed 3 features per sample

#### **Extended Features (Used in Some Experiments):**

**Option A - Basic Statistical Features:**

```python
features = [
    "mean_hr",      # Average HR
    "std_hr",       # Standard deviation (variability measure)
    "min_hr",       # Minimum HR value
    "max_hr",       # Maximum HR value
    "range_hr"      # max_hr - min_hr
]
```

**Shape:** `(n_samples, 5)`

**Option B - With Delta Features (Emotion Comparison):**

```python
features = [
    "mean_hr", "std_hr", "min_hr", "max_hr", "range_hr",
    "delta_mean",   # mean_hr_angry - mean_hr_sad
    "delta_std",    # std_hr_angry - std_hr_sad
    "delta_range"   # range_hr_angry - range_hr_sad
]
```

**Shape:** `(n_samples, 8)` - Includes difference between emotional states

**Option C - Emotion-Specific Features:**

```python
features = [
    "mean_hr_sad", "std_hr_sad", "range_hr_sad", "min_hr_sad",      # Sad video stats
    "mean_hr_angry", "std_hr_angry", "range_hr_angry", "min_hr_angry",  # Angry video stats
    "delta_mean", "delta_std", "delta_range"                         # Differences
]
```

**Shape:** `(n_samples, 11)` - Paired emotion comparison

---

### ❌ **Features NOT Used:**

| Feature Type                     | Status      | Reason                                        |
| -------------------------------- | ----------- | --------------------------------------------- |
| **Raw HR Array**                 | ❌ Not Used | Time series converted to statistical features |
| **Heart Rate Variability (HRV)** | ❌ Not Used | RMSSD, pNN50, etc. not calculated             |
| **Frequency Domain (FFT)**       | ❌ Not Used | No spectral analysis performed                |
| **Time Series Features**         | ❌ Not Used | No LSTM/sequential modeling                   |
| **Video-Specific Features**      | ⚠️ Implicit | Video ID used for grouping, not as feature    |

---

## 🔧 Question 2: Feature Preprocessing

### **Normalization/Standardization Method:**

#### **StandardScaler (Z-score Normalization)**

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(train[features])
x_train_scaled = scaler.transform(train[features])
x_val_scaled = scaler.transform(val[features])
x_test_scaled = scaler.transform(test[features])
```

**Formula:**

```
x_scaled = (x - μ) / σ

where:
  μ = mean of feature in training set
  σ = standard deviation of feature in training set
```

**Key Points:**

- ✅ Fitted **only on training data** (prevents data leakage)
- ✅ Same scaler applied to validation and test sets
- ✅ **Critical:** Production API must use the same scaler saved during training

---

### **Scaling Parameters (Must Be Saved):**

The scaler learns these from training data:

```python
scaler.mean_      # Mean of each feature [μ₁, μ₂, μ₃]
scaler.scale_     # Std dev of each feature [σ₁, σ₂, σ₃]
```

**Example for 3-feature model:**

```python
# Hypothetical values (actual values depend on your data)
scaler.mean_  = [78.5, 6.8, 42.3]     # [mean_hr, std_hr, range_hr]
scaler.scale_ = [12.4, 2.1, 18.7]     # Standard deviations
```

**Saving/Loading:**

```python
import joblib

# After training
joblib.dump(scaler, "scaler.pkl")

# In production (connecting.py)
scaler = joblib.load("scaler.pkl")
```

---

### **Missing Value Handling:**

#### **Current Approach: ⚠️ No Explicit Handling**

Based on the code review:

- No `fillna()` or imputation detected
- Assumes data is **pre-cleaned** during CSV generation
- Raw data shows some empty cells (e.g., `85,,69,99`) but these appear to be handled upstream

#### **Recommendation for Production:**

Add defensive checks in API:

```python
@app.post("/predict")
def predict(data: EmotionInput):
    # Validate input
    if any(pd.isna([data.mean_hr, data.std_hr, data.range_hr])):
        raise HTTPException(400, "Missing required features")

    if data.mean_hr <= 0 or data.std_hr < 0 or data.range_hr < 0:
        raise HTTPException(400, "Invalid heart rate values")

    # Continue with prediction...
```

---

## 📐 Question 3: Input Shape

### **✅ Aggregated Features Only (Fixed Shape)**

**Production Model Input:**

```python
Input Shape: (1, 3)  # Single prediction
            # [mean_hr, std_hr, range_hr]

Batch Input Shape: (n_samples, 3)  # Multiple predictions
```

**Example:**

```python
# Single sample
x = np.array([[85.5, 6.2, 45.0]])  # Shape: (1, 3)

# Multiple samples
x = np.array([
    [85.5, 6.2, 45.0],
    [72.3, 4.1, 38.0],
    [91.2, 7.8, 52.0]
])  # Shape: (3, 3)
```

---

### **❌ NOT Variable Length / Time Series**

The model does **NOT** accept:

- ❌ Raw HR arrays: `[83, 83, 80, 99, 85, ...]`
- ❌ Variable length sequences
- ❌ Padded time series
- ❌ RNN/LSTM-style inputs with shape `(batch, timesteps, features)`

---

### **Data Flow: Raw → Features → Model**

```
┌─────────────────────────────────┐
│   Raw Heart Rate Time Series   │
│   [83, 83, 80, 99, 85, 80, ...]│
│   Shape: (n_timesteps,)         │
└─────────────┬───────────────────┘
              │
              │ Feature Extraction (done offline)
              │ - Calculate mean, std, min, max, range
              │
┌─────────────▼───────────────────┐
│   Statistical Features          │
│   [mean_hr, std_hr, range_hr]  │
│   Shape: (3,)                   │
└─────────────┬───────────────────┘
              │
              │ Standardization (scaler.transform)
              │
┌─────────────▼───────────────────┐
│   Scaled Features               │
│   [(78-μ₁)/σ₁, (6-μ₂)/σ₂, ...] │
│   Shape: (3,)                   │
└─────────────┬───────────────────┘
              │
              │ Model Prediction
              │
┌─────────────▼───────────────────┐
│   Emotion Probabilities         │
│   [p_sad=0.38, p_angry=0.62]   │
└─────────────────────────────────┘
```

---

## 🎯 Production API Requirements

### **Required Files:**

1. ✅ `model_gb.pkl` - Trained Gradient Boosting model
2. ✅ `scaler.pkl` - Fitted StandardScaler with training statistics
3. ✅ `thresholds.json` - Optimized thresholds for 3-class classification

### **Input Requirements:**

```json
{
  "mean_hr": 85.5, // Must be > 0 (beats per minute)
  "std_hr": 6.2, // Must be >= 0
  "range_hr": 45.0, // Must be >= 0
  "mbti_tf": "t" // Must be "t" or "f" (case insensitive)
}
```

### **Output Format:**

```json
{
  "prob_angry": 0.6234,
  "prob_sad": 0.3766,
  "class_id": 2,
  "class_name": "angry",
  "color_hex": "#F44336",
  "emoji": "😠",
  "mbti_group": "T",
  "threshold_used": {
    "th_angry": 0.45,
    "th_sad": 0.61
  }
}
```

---

## 📝 Summary Table

| Aspect               | Details                                                |
| -------------------- | ------------------------------------------------------ |
| **Feature Type**     | Statistical aggregates (mean, std, range)              |
| **Feature Count**    | 3 (production) / 5-11 (experiments)                    |
| **Input Shape**      | `(n_samples, 3)` fixed                                 |
| **Preprocessing**    | StandardScaler (Z-score normalization)                 |
| **Missing Values**   | Assumed pre-cleaned (no explicit handling)             |
| **Temporal Info**    | ❌ Not preserved (aggregated)                          |
| **HRV Features**     | ❌ Not used                                            |
| **Frequency Domain** | ❌ Not used                                            |
| **Model Type**       | Traditional ML (SVM, Random Forest, Gradient Boosting) |

---

## 🔬 Feature Extraction Process (Inferred)

The preprocessed CSV files were likely created using this logic:

```python
def extract_hr_features(hr_array):
    """
    Extract statistical features from raw heart rate array

    Args:
        hr_array: numpy array of heart rate values (e.g., from video session)

    Returns:
        dict of features
    """
    return {
        'mean_hr': np.mean(hr_array),
        'std_hr': np.std(hr_array),
        'min_hr': np.min(hr_array),
        'max_hr': np.max(hr_array),
        'range_hr': np.max(hr_array) - np.min(hr_array)
    }

# Example usage:
# video_hr = [83, 83, 80, 99, 85, 80, 78, 114, 85, 85, 71, 68, 66]
# features = extract_hr_features(video_hr)
# >> {'mean_hr': 82.69, 'std_hr': 13.08, 'min_hr': 66, 'max_hr': 114, 'range_hr': 48}
```

---

## 🚨 Important Notes for Flutter Integration

When sending data from Flutter to the API:

1. **Ensure HR features are pre-calculated** - Don't send raw arrays
2. **Match the exact 3 features** used in production: `[mean_hr, std_hr, range_hr]`
3. **The API handles scaling internally** - Send raw statistical values
4. **MBTI type is required** - Default to "t" if unknown

```dart
// Flutter example
final response = await http.post(
  url,
  body: jsonEncode({
    'mean_hr': 85.5,      // Already calculated
    'std_hr': 6.2,        // Already calculated
    'range_hr': 45.0,     // Already calculated
    'mbti_tf': 't'
  }),
);
```

---

**Last Updated:** October 30, 2025  
**Model Version:** Gradient Boosting with TF-aware thresholds  
**Feature Set:** 3-feature minimal (mean_hr, std_hr, range_hr)
