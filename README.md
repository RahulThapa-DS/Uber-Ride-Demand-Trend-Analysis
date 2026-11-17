# 🚗 Uber Ride Demand & Trend Analysis

An end-to-end Data Science and Machine Learning project analyzing 2016 Uber trip logs, temporal demand fluctuations, cyclical feature engineering, trip duration modeling, and 4-hour block surge forecasting using Python.

---

## 📌 1. Executive Summary & Dataset Overview

This project processes and analyzes a full-year operational dataset of **1,155 raw Uber trip records** spanning **January 1, 2016 to December 31, 2016** (`1,154` valid individual trips after removing the dataset summary footer, totaling **12,204.7 miles** tracked).

### Dataset Schema & Features
* **Raw Features:** `START_DATE*`, `END_DATE*`, `CATEGORY*` (`Business` vs. `Personal`), `START*`, `STOP*`, `MILES*`, and `PURPOSE*` (e.g., *Meeting*, *Meal/Entertain*, *Errand/Supplies*, *Customer Visit*).
* **Derived Operational Metrics:**
  * `Duration_Mins`: Exact trip length computed from parsed `START_DATE*` and `END_DATE*` timestamps.
  * `Speed_MPH`: Effective trip velocity (`MILES / (Duration_Mins / 60)`).
  * `Time_Block`: Six 4-hour operational windows per day (`0` through `5`, mapping `Request Hour // 4`).
  * `Is_Peak_Block`: Binary indicator capturing peak daytime/evening demand blocks (`2`, `3`, and `4`).

---

## 🛠️ 2. Data Science Implementation & Tech Stack

* **Language:** Python 3.x
* **Data Manipulation:** `pandas`, `numpy`
* **Visualization:** `matplotlib`, `seaborn`
* **Machine Learning & Evaluation:** `scikit-learn` (`LinearRegression`, `LogisticRegression`, `RandomForestRegressor`, `RandomForestClassifier`, `ExtraTreesClassifier`, `GradientBoostingClassifier`, `StandardScaler`, `make_pipeline`, `average_precision_score`, `precision_recall_curve`, `balanced_accuracy_score`), `xgboost` (`XGBClassifier`)
* **Environment:** JupyterLab / Jupyter Notebook (`Uber Ride Demand & Trend Analysis.ipynb`)

---

## ⚙️ 3. Data Preprocessing & Feature Engineering Pipeline

To prevent target leakage and solve extreme zero-inflation in sparse ride logs, raw timestamps are processed through a strict chronological pipeline:

### A. Data Cleansing & Temporal Grid Construction
1. **Timestamp Standardization & Cleaning:** Standardized column names, parsed mixed datetime formats, handled missing `PURPOSE*` entries, and stripped the non-trip summary row (`1,155` → `1,154` trips).
2. **Continuous Time-Grid Resampling (`pd.MultiIndex.from_product`):**
   * **1-Hour Baseline Grid (`8,784` intervals):** Filling inactive hours with `0` rides revealed **88.5% zero-ride sparsity**, causing standard regression models to under-predict rare demand spikes.
   * **4-Hour Operational Block Grid (`2,196` intervals):** Aggregating into 4-hour blocks reduced zero-ride sparsity from **88.5% down to 69.6%** and established a data-driven **Surge Window** definition (`>= 2 rides / 4-hr block`, yielding a `12.2%` surge rate in training and `18.9%` in testing).

### B. Feature Engineering (`16` Time-Series Predictors)
* **Calendar Indicators:** `Time_Block`, `Day_Of_Week`, `Month`, `Is_Weekend`, and `Is_Peak_Block`.
* **Cyclical Trigonometric Encoding (`add_calendar_features`):** Applied $\sin(2\pi \cdot t / T)$ and $\cos(2\pi \cdot t / T)$ (`Period_Sin`, `Period_Cos`, `Day_Sin`, `Day_Cos`) so models preserve circular continuity across midnight (`Block 5` → `Block 0`) and weekend boundaries (`Sunday` → `Monday`).
* **Leakage-Free Autoregressive Lags & Rolling Windows:**
  * `Demand_Lag_1blk` & `Demand_Lag_2blk` (4-hour and 8-hour momentum).
  * `Demand_Lag_6blk_24h` & `Demand_Lag_42blk_7d` (same 4-hour window 24 hours and 7 days prior).
  * `Rolling_Mean_3blk_12h`, `Rolling_Mean_6blk_24h`, and `Rolling_Max_6blk_24h` — strictly shifted by 1 step via `.shift(1)` prior to `.rolling()` so current-window ride counts never leak into predictors.

---

## 🤖 4. Exploratory Analysis & Applied Machine Learning Models

### Exploratory Data Analysis (EDA) Highlights
* **Category & Purpose Dominance:** Business travel accounts for the overwhelming majority of tracked mileage and trip volume, concentrated around client visits, meetings, and meal/entertainment windows.
* **Temporal Concentration:** Demand exhibits strong diurnal seasonality, peaking across afternoon and evening operational blocks (`Time_Block` `2–4`, covering 08:00–20:00) while overnight blocks (`00:00–04:00`) remain near zero.
* **Right-Skewed Trip Durations:** Most trips cluster under 25 minutes, with long-distance outlier trips creating heavy right skew—motivating log-target transformation (`np.log1p`) for duration regression.

### Multi-Stage Modeling Architecture (Chronological 80/20 Split)
All time-series models use a strict chronological split (**80% Train** = `1,756` blocks / **20% Test** = `440` blocks) with an internal validation slice (`val_split`) to optimize the F1 decision threshold (`tune_f1_threshold`) rather than relying on a default `0.50` cutoff.

#### 1. Step 4 — Trip-Level Duration Prediction (Log-Target Regression)
* **Task:** Predict individual trip duration (`Duration_Mins`) across `1,154` trips (`30` encoded features) and flag high-duration trips (`>= 24.03 mins`).
* **Winning Model:** **`Linear Regression [Log-Target]`** achieved **`R² = 60.72%`**, **`RMSE = 10.84 mins`**, and **`F1 = 56.25%`**, outperforming tree regressors by capturing multiplicative distance-to-duration scaling without overfitting outlier trips.

#### 2. Step 5 — 4-Hour Block Surge Classification (Ranked by PR-AUC)
Because surge windows represent a minority class (`12.2%` train), models are ranked primarily by **PR-AUC** (Precision-Recall Area Under the Curve) alongside **Balanced Accuracy** and **Surge F1-Score**:

| Rank | Model | PR-AUC (Primary) | ROC-AUC | Balanced Accuracy | Surge F1-Score | Precision | Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Logistic Regression (Balanced)** | **0.4174** | **0.8057** | **67.82%** | **46.56%** | **41.51%** | **53.01%** |
| **2** | Extra Trees (Balanced) | 0.4068 | 0.7852 | 69.24% | 47.22% | 38.35% | 61.45% |
| **3** | Random Forest (Balanced) | 0.4028 | 0.7849 | 69.70% | 47.71% | 38.52% | 62.65% |
| **4** | Gradient Boosting (Regularized) | 0.3500 | 0.7608 | 62.68% | 39.33% | 36.84% | 42.17% |
| **5** | XGBoost (Scale-Pos-Weighted) | 0.3332 | 0.7477 | 58.18% | 32.37% | 31.11% | 33.73% |

#### 3. Step Final — Architectural Validation: 1-Hour Regression vs. 4-Hour Classification

| Pipeline Architecture | Grid Resolution | Zero-Slot Sparsity | Selected Model | PR-AUC (vs Baseline) | Balanced Accuracy | Surge F1-Score |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: |
| **1. Hourly Demand Regression (Baseline)** | `8,784` hrs (1-hr) | 88.5% | Random Forest Regressor | `0.3497` (vs `0.1537`) | 66.26% | 39.89% |
| **2. 4-Hour Block Surge Classifier (Final)** | `2,196` blocks (4-hr) | **69.6%** | **Logistic Regression (Balanced)** | **`0.4174`** (vs `0.1886`) | **67.82%** | **46.56%** |

---

## 📊 5. Key Results, Challenges & Business Applications

### Summary of Findings
* **2.2× Lift Over No-Skill Baseline:** `StandardScaler` + `LogisticRegression(class_weight="balanced")` with a tuned probability threshold (`>= 0.667`) achieved a **PR-AUC of `0.4174`** (vs. `0.1886` baseline) and **ROC-AUC of `0.8057`**.
* **+6.67% Surge F1 Gain from Temporal Aggregation:** Shifting from 1-hour Random Forest regression (`39.89%` F1) to 4-hour balanced classification (`46.56%` F1) cut zero-slot sparsity by **18.9 percentage points** and boosted alert precision from `31.99%` to `41.51%`.
* **Primary Surge Drivers:** Standardized log-odds coefficients confirm that **24-hour rolling demand (`Rolling_Mean_6blk_24h` / `Rolling_Max_6blk_24h`)**, **recent 1-block lag (`Demand_Lag_1blk`)**, and **cyclical time-of-day encodings (`Period_Sin`, `Period_Cos`, `Is_Peak_Block`)** are the strongest predictors of upcoming demand spikes.

### Challenges & Future Improvements
* **Data Sparsity & Single-User Scope:** Because the log captures `1,154` trips across an entire year, granular 1-hour intervals remain heavily zero-inflated (`88.5%`). Expanding to fleet-wide multi-driver logs would stabilize hourly variance.
* **Missing Exogenous Signals:** Incorporating **hourly weather conditions** (precipitation, temperature extremes), **local event/holiday calendars**, and **spatial pickup/dropoff zones** would substantially reduce false negatives during unscheduled demand spikes.
* **Pricing & Dynamic Dispatch Integration:** While this dataset tracks mileage and trip duration rather than fare amounts, pairing predicted surge probabilities (`>= 0.667`) with dynamic surge-multiplier tiers enables proactive driver repositioning 4 hours ahead of peak windows.

---

## 📁 6. Repository Structure

```text
├── Uber Ride Demand & Trend Analysis.ipynb   # Complete end-to-end EDA, feature engineering & ML notebook
├── UberDataset.csv                           # Raw 2016 Uber trip log dataset (1,155 rows)
└── README.md                                 # Project documentation, methodology & evaluation report
```


## 🚀 How to Run

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/RahulThapa-DS/Uber-Ride-Analysis.git](https://github.com/RahulThapa-DS/Uber-Ride-Analysis.git)
   cd Uber-Ride-Analysis
   ```
2. **Install required packages:**

```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn
```
3. **Launch the notebook:**

```bash
jupyter notebook "Uber Ride Analysis -Project 2.ipynb"
```
---
## 📋 Project Deliverables Checklist

### 1. Data Cleaning & Standardization
- [x] Load dataset (`Uber Drives - .csv`) and strip non-trip summary rows.
- [x] Parse datetime values for `START_DATE*` and `END_DATE*`.
- [x] Handle missing values in trip attribute columns.

### 2. Temporal & Lag Feature Engineering
- [x] Extract start and end hour, day of week, and date fields.
- [x] Compute duration in minutes (`Duration_mins`).
- [x] Map hourly time slots (`Early Morning`, `Morning Peak`, `Afternoon`, `Evening Peak`, `Late Night`).
- [x] Create indicator flags (`Is_Rush_Hour`, `Is_Night`).
- [x] Construct cyclical features (`hour_sin`, `hour_cos`).
- [x] Generate lag and rolling mean demand metrics (`Demand_Lag_1h`, `Demand_Lag_24h`, `Rolling_Mean_3h`).

### 3. Exploratory Visualization
- [x] Generate hourly request volume distribution plot.
- [x] Generate category/purpose breakdown per time slot heatmap.
- [x] Generate log-scaled mileage distribution histogram.
