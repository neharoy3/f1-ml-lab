# **F1 Lap Time Predictor**

## **1. Purpose**

Predict Formula 1 lap times using machine learning. Built as a learning exercise to understand regression models, data cleaning, feature engineering, and model deployment. The goal is not perfect predictions but understanding *why* models make certain decisions and *how* F1 data behaves.

---

## **2. Problem Definition**

**Type:** Regression (predicting a continuous number - seconds)

**Input variables:**
- Lap number
- Tyre age (laps since new tyres)
- Stint number
- Race position
- Driver
- Tyre compound

**Output:** Lap time in seconds

**Why this matters:** Lap time depends on non-linear factors - tyres degrade, fuel burns off, track grip evolves. Simple models fail. This makes F1 a perfect regression case study.

---

## **3. Data Source**

**FastF1** - Open-source F1 data API

**Why FastF1 over alternatives:**
- Free (Ergast API deprecated)
- Includes sector times and tyre data
- Actively maintained
- Works in Colab with no setup

Data collected from user-specified race (season + round number input)

---

## **4. Methodology - Step by Step**

### **Step 1: Data Loading**

```python
session = fastf1.get_session(season, round_num, 'R')
session.load(laps=True)
```

**Why 'R' (Race):** Qualifying is one-lap pace, race has tyre degradation patterns that make prediction interesting.

### **Step 2: Data Cleaning**

**Removed:**
- Lap 1 (standing start, collisions, unusual)
- Laps with pit stops (`PitInTime` not null)
- Invalid lap times (NaT values)
- Outliers beyond 1.5 IQR

**Why IQR instead of fixed thresholds:** Lap times vary by circuit (Monaco 75s, Spa 105s). IQR adapts automatically.

**Why remove pit laps:** Pit entry/exit laps are 5-10 seconds slower. Including them confuses the model - it learns "lap 12 is slow" instead of "laps with pit stops are slow."

### **Step 3: Feature Engineering**

**Features created:**
- `LapTimeSeconds`: Target variable (converted from timedelta)
- `DriverCode`: Encoded as integers
- `Compound`: SOFT/MEDIUM/HARD → 0/1/2

**Why no fuel load feature:** Fuel burns linearly (~0.05s per lap). Lap number captures this effect indirectly. Adding explicit fuel didn't improve accuracy.

**Why no sector times initially:** Sector data often missing in FastF1. Keeping features that always exist makes model robust across races.

### **Step 4: Model Selection**

**Tested models:**

| Model | Train MAE | Test MAE | Gap | Verdict |
|-------|-----------|----------|-----|---------|
| Linear Regression | 0.89s | 0.91s | 0.02s | Poor - can't handle non-linear |
| Decision Tree (unpruned) | 0.01s | 0.52s | 0.51s | Overfitted - memorized |
| Decision Tree (max_depth=5) | 0.31s | 0.35s | 0.04s | Better but still worse |
| Random Forest (100 trees) | 0.23s | 0.27s | 0.04s | **Winner** |

**Why Random Forest won:**

1. **Handles non-linearity naturally** - Tyre degradation isn't straight line. Lap 1-5: slow (cold tyres). Lap 6-20: fast (optimal). Lap 21+: slow (worn tyres). Random Forest captures this U-shape. Linear regression can't.

2. **No feature scaling needed** - Lap numbers (1-70) and driver codes (0-19) have different scales. Neural networks require normalization. Random Forest doesn't care.

3. **Resists overfitting** - Averages 100 trees, each trained on different data subsets. Decision Tree memorizes specific laps ("Lap 23, VER = 89.4s"). Random Forest learns general patterns ("VER is 0.2s faster than HAM").

4. **Provides feature importance** - Tells me what matters:
   - TyreLife: 45% importance
   - LapNumber: 30% importance
   - DriverCode: 15% importance
   - Compound: 10% importance

   This validated my domain knowledge - tyres matter most.

**Why not Neural Networks:** Would work but needs more data (1000+ laps). One race = 200 laps. Random Forest performs better on small datasets.

### **Step 5: Training Setup**

```python
X_train, X_test, y_train, y_test = train_test_split(0.2)
model = RandomForestRegressor(n_estimators=100, max_depth=10)
```

**Why 80/20 split:** Standard practice. Enough training data (160 laps), enough test data (40 laps) for reliable evaluation.

**Why max_depth=10:** Default (None) overfits. Depth 10 balances underfitting vs overfitting on 200 samples.

**Why n_estimators=100:** More trees = better but slower. 100 is sweet spot for this dataset size.

### **Step 6: Evaluation Metrics**

**Used:**
- **MAE (Mean Absolute Error):** 0.27 seconds. Average prediction misses by quarter second.
- **R² (Coefficient of Determination):** 0.89. Model explains 89% of lap time variance.

**Why MAE over RMSE:** MAE is interpretable ("off by 0.27s on average"). RMSE penalizes large errors more, but F1 rarely has huge outliers after cleaning.

**What 0.27s MAE means in F1 terms:**
- Good enough to know which driver is faster
- Not good enough to predict exact qualifying gaps (often 0.1s)
- Acceptable for a beginner project

---

## **5. Analysis of Results**

### **What the model learned correctly:**

1. **Tyre age matters most** - 45% importance. Confirms F1 reality: tyres lose 0.1-0.2s per lap.

2. **Lap 15-25 are fastest** - Predicted times drop then rise. Matches tyre warm-up then degradation.

3. **Driver skill visible** - VER consistently 0.15s faster than average. PER 0.05s slower. Reasonable.

### **Where the model fails:**

1. **Safety car periods** - Not in training data. Would predict normal lap times during SC (wrong).

2. **Weather changes** - No temperature data. Rain would break predictions.

3. **Track evolution** - Doesn't know grip increases over weekend. Qualifying would be faster than practice.

### **Error patterns observed:**

- **Lap 1 still slightly off** (even after removal) - Some lap 1 data remains in other laps' tyre life.
- **First lap after pit stop** - Slightly overpredicts (cold tyres effect weaker than model thinks).

---

## **6. Limitations & Known Issues**

| Limitation | Why it exists | How to fix |
|------------|---------------|-------------|
| One race only | Model doesn't generalize across circuits | Train on multiple races |
| No weather data | Rain changes everything | Add weather API |
| No fuel correction | Fuel burning improves times | Calculate fuel laps remaining |
| Driver encoding arbitrary | VER=1, HAM=2 has no meaning | One-hot encoding |
| No track evolution | Grip increases over weekend | Add session number feature |

---

## **7. What I Learned**

**Machine Learning:**
- Random Forest is beginner-friendly for a reason - it just works
- Cleaning matters more than model choice
- Feature importance is underrated (teaches domain knowledge)
- Overfitting is real and visible (Decision Tree trained to 0.01s MAE, tested at 0.52s)

**F1 Domain:**
- Tyres are 45% of lap time prediction
- Driver matters less than I thought (only 15%)
- Lap 1 is useless for training
- Pit laps break patterns completely

**Production:**
- Streamlit makes UI trivial
- .pkl files save hours of retraining
- GitHub + Streamlit Cloud = free deployment

---

## **8. Conclusion**

This project successfully predicts F1 lap times within 0.27 seconds using a Random Forest model on basic race data. The model correctly identifies tyre age as the dominant factor and captures non-linear degradation patterns that linear models miss.

**Successes:**
- Working prediction with acceptable error
- Clean, documented code
- Deployed UI accessible online
- Understanding of *why* Random Forest works here

**Failures (learning opportunities):**
- Initial Decision Tree overfitting
- Missing sector times broke first model
- Forgetting feature order when loading .pkl

**Final verdict:** A solid beginner project that demonstrates understanding of regression workflows, data cleaning, model selection, and deployment. Ready for GitHub and for adding more F1 ML experiments.

---

## **9. References**

- FastF1 Documentation: https://docs.fastf1.dev/
- Scikit-learn Random Forest: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
- Streamlit Docs: https://docs.streamlit.io/
- F1 tyre degradation analysis: https://www.f1technical.net/articles/tyre-degradation

---
*by Neha Roy*