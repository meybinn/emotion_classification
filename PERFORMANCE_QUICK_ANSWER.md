# Quick Answers: Performance & Deployment

## ⏱️ Recording Duration

**Minimum:** 30-60 seconds (low confidence)  
**Recommended:** 2-3 minutes (moderate confidence)  
**Optimal:** 5+ minutes (high confidence)

**Why?** Statistical features need sufficient samples. Training used ~5-10 minute videos.

---

## 📡 Sampling Rate

**Can handle variable rates!** ✅

Model uses aggregated features (mean, std, range), not time series.

**Recommended:** 1-5 Hz  
**Acceptable:** 0.5-10 Hz  
**Overkill:** >10 Hz (no benefit)

---

## 🧹 Data Quality

**Current training:** No explicit filtering ⚠️

**Recommended for production:**

- ✅ Remove outliers (HR < 40 or > 220 bpm)
- ✅ Apply moving average (5-sample window)
- ✅ Detect flatlines (sensor disconnection)
- ✅ Check for excessive noise (std > 50)

---

## 📊 Training Accuracy

| Metric            | Value  | Notes                      |
| ----------------- | ------ | -------------------------- |
| **Validation F1** | 67-75% | Varies by personality type |
| **T-group**       | 75%    | Better for Thinking types  |
| **F-group**       | 66.7%  | Lower for Feeling types    |
| **Test Accuracy** | 70-80% | Realistic expectation      |

**⚠️ Limitations:**

- Small dataset (16 patients, 32 videos)
- High variance due to limited data
- Some models show overfitting (RF: 100%)

---

## 🎯 Confidence Thresholds

| Confidence    | Action                              |
| ------------- | ----------------------------------- |
| **≥ 0.70**    | Show in UI, reliable classification |
| **0.50-0.70** | Moderate - use with caution         |
| **0.30-0.50** | Low - log only, don't display       |
| **< 0.30**    | Unreliable - discard                |

**Probability Zones:**

- `0.00-0.45`: SAD
- `0.45-0.61`: NORMAL (low confidence zone)
- `0.61-1.00`: ANGRY

---

## ⚠️ Edge Cases

### **1. Very Short Recordings**

```python
if samples < 30:
    return {"error": "insufficient_data"}
elif samples < 120:
    reduce_confidence_by(30%)
```

### **2. Noisy Data**

Check for:

- Flatlines (sensor stuck/disconnected)
- Excessive spikes (>50 bpm between readings)
- Too many outliers (>20% of data)

**Action:** Return error, ask user to retry

### **3. Extreme Heart Rates**

Training range: 59-138 bpm (mean), 42-176 bpm (min-max)

If outside range → Add warning: "Outside training distribution"

### **4. Missing MBTI**

Default to `"all"` threshold (F1 = 66.7%)

---

## 🚀 Production Checklist

- [ ] Implement minimum duration check (120+ samples)
- [ ] Add outlier filtering (40-220 bpm)
- [ ] Detect noisy signals (flatlines, spikes)
- [ ] Check distribution shift (out-of-training-range)
- [ ] Return confidence score with prediction
- [ ] Add warnings for low-quality data
- [ ] Implement MBTI fallback to "all"
- [ ] Set UI display threshold at 0.70+ confidence

---

## 📱 Flutter Integration Tips

```dart
// Before sending to API
if (hrReadings.length < 30) {
  showError("Need at least 30 seconds of data");
  return;
}

// Calculate features
double meanHr = calculateMean(hrReadings);
double stdHr = calculateStd(hrReadings);
double rangeHr = calculateRange(hrReadings);

// Quality check
if (stdHr > 50) {
  showWarning("Signal quality may be poor");
}

// Send to API
final result = await api.predict(meanHr, stdHr, rangeHr, mbtiType);

// Check confidence before displaying
if (result.confidence >= 0.70) {
  showEmotionResult(result);
} else {
  showLowConfidenceMessage();
}
```

---

## 🎯 Expected Real-World Performance

| Scenario               | Accuracy | Notes                           |
| ---------------------- | -------- | ------------------------------- |
| **Optimal conditions** | 75-80%   | 5+ min recording, good sensor   |
| **Typical usage**      | 65-75%   | 2-3 min recording, normal noise |
| **Poor conditions**    | 50-60%   | Short recording or noisy data   |
| **Edge cases**         | <50%     | Return error instead            |

**Bottom line:** This is a research-grade model suitable for non-critical applications (wellness apps, mood tracking), NOT for medical diagnosis.

---

**Key Takeaway:** The model performs best with 5+ minute recordings at 1-5 Hz sampling. Always check data quality and confidence scores before displaying results to users.
