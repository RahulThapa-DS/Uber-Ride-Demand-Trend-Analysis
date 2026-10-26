# 🚗 Uber Ride Demand & Trend Analysis

An end-to-end Data Science and Machine Learning project analyzing Uber trip patterns, temporal demand fluctuations, feature engineering, and predictive demand forecasting using Python.

---

## 📌 Executive Summary & Project Overview

This project processes and analyzes a dataset of 1,155 Uber trip records (1,154 individual trips totaling 12,204.7 miles tracked). The analysis transforms raw start/end timestamps into granular temporal features, categorizes trips into distinct operational time slots, evaluates distance distributions, and builds predictive models for hourly demand.

### Key Objectives & Operational Insights
* **Data Cleansing & Standardization:** Handle missing values, parse non-uniform datetime entries (`START_DATE*`, `END_DATE*`), and remove dataset summary rows.
* **Feature Engineering:** Derive temporal metrics, cyclical transformations, time-slot categories, and time-series lag/rolling features.
* **Exploratory Analysis:** Uncover hourly ride demand, peak operational windows, trip distance distributions, and purpose/category patterns.
* **Demand Forecasting:** Evaluate hourly ride counts using Scikit-Learn regression models.

---

## 🛠️ Data Science Implementation & Stack

* **Programming Language:** Python 3.x
* **Data Manipulation:** `pandas`, `numpy`
* **Data Visualization:** `matplotlib`, `seaborn`, `plotly`
* **Machine Learning:** `scikit-learn` (`RandomForestRegressor`, `train_test_split`, `mean_squared_error`, `r2_score`)
* **Environment:** Jupyter Notebook / JupyterLab

---

## ⚙️ Data Preprocessing & Feature Engineering Pipeline

1. **Data Preprocessing & Cleaning:**
   * Ingests raw trip data (`Uber Drives - .csv`).
   * Removes non-trip summary/total rows (e.g., index 1155).
   * Converts `START_DATE*` and `END_DATE*` columns into standard datetime objects.
   * Cleans missing values in categorical metadata (e.g., `PURPOSE*`).

2. **Feature Engineering Strategy:**
   * **Temporal Metrics:** `Request Hour`, `Request Day`, `Request Date`, `End Hour`, `End Day`, `End Date`.
   * **Trip Metrics:** `Duration_mins` (calculated trip duration from timestamp deltas).
   * **Operational Time Slots:**
     * **Early Morning:** 04:00 – 07:59
     * **Morning Peak:** 08:00 – 11:59
     * **Afternoon:** 12:00 – 16:59
     * **Evening Peak:** 17:00 – 21:59
     * **Late Night:** 22:00 – 03:59
   * **Binary Indicators:** `Is_Rush_Hour` (`08-10`, `17-19`), `Is_Night` (`22-05`).
   * **Cyclical Encoding:** Sine and cosine transformations (`hour_sin`, `hour_cos`) to capture 24-hour periodicity.
   * **Time-Series Lag Features:** `Demand_Lag_1h`, `Demand_Lag_24h`, `Rolling_Mean_3h`.

---

## 📊 Exploratory Data Analysis & Insights

### Key Findings
* **Peak Demand Hours:** Ride requests peak heavily during standard commuting windows—specifically **Morning Peak (8 AM – 11 AM)** and **Evening Peak (5 PM – 8 PM)**.
* **Trip Distances:** The distribution of `MILES*` is right-skewed, showing that the majority of urban rides are short-distance trips under 10 miles.
* **Ride Categories:** Business rides dominate weekday peak slots, whereas Personal rides lean toward evenings and weekends.

### Visualizations Generated
* **Hourly Ride Request Volume:** A countplot displaying total ride requests segmented by `Request Hour`.
* **Time Slot vs. Purpose/Category Heatmap:** A cross-tabulation heatmap illustrating trip volume across time slots and business/personal purposes.
* **Trip Distance Distribution:** A histogram with KDE overlay showing trip distances (`MILES*`) on a log scale.

---

## 🤖 Machine Learning & Demand Forecasting

A **Random Forest Regressor** was implemented to forecast hourly ride volume based on temporal attributes.

```python
# Model Setup
X = hourly_df[['Request Hour', 'Request Day']]
y = hourly_df['Ride_Count']

# Performance Metrics
RMSE: 0.36
R² Score: -0.02
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
