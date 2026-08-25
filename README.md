# 🚗 Uber Ride Demand & Trend Analysis

An end-to-end Data Science project analyzing Uber ride patterns, temporal demand fluctuations, trip characteristics, and predictive forecasting using Python.

---

## 📌 Project Overview
This repository contains data preprocessing, exploratory data analysis (EDA), visual statistics, and machine learning models designed to extract business insights from historical Uber trip data.

### Key Objectives
* **Data Cleansing & Transformation:** Handle missing values, parse datetime entries (`START_DATE*`, `END_DATE*`), and derive engineered metrics.
* **Exploratory Analysis:** Uncover hourly ride demand, peak operational windows, trip distance distributions, and purpose/category patterns.
* **Demand Forecasting:** Predict ride counts across specific hours and days using Scikit-learn regression models.

---

## 🛠️ Data Science Implementation & Stack

* **Language:** Python 3.x
* **Data Processing:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn (`RandomForestRegressor`, `train_test_split`, `mean_squared_error`, `r2_score`)

---

## ⚙️ Data Preprocessing & Feature Engineering

1. **Datetime Conversion:** Standardized non-uniform `START_DATE*` and `END_DATE*` columns into native Python datetimes.
2. **Missing Value Handling:** Cleaned missing values in categorical metadata (e.g., `PURPOSE*`).
3. **Feature Extraction:**
   * `Request Hour`: Extracted hour of day (0–23).
   * `Request Day`: Extracted day name (Monday–Sunday) and mapped to numerical codes.
   * `Time Slot`: Categorized hours into operational windows (`Early Morning`, `Morning Peak`, `Afternoon`, `Evening Peak`, `Late Night`).
   * `Duration_mins`: Calculated exact trip duration from start/end timestamp deltas.

---

## 📊 Exploratory Data Analysis & Insights

### Key Findings
* **Peak Demand Hours:** Ride requests peak heavily during standard commuting windows—specifically **Morning Peak (8 AM – 11 AM)** and **Evening Peak (5 PM – 8 PM)**.
* **Trip Distances:** The distribution of `MILES*` is right-skewed, showing that the majority of urban rides are short-distance trips under 10 miles.
* **Ride Categories:** Business rides dominate weekday peak slots, whereas Personal rides lean toward evenings and weekends.

---

## 🤖 Machine Learning & Modeling

A **Random Forest Regressor** was built to forecast hourly ride volume based on temporal features (`Request Hour`, `Request Day`).

```python
# Model Setup
X = hourly_df[['Request Hour', 'Request Day']]
y = hourly_df['Ride_Count']

# Performance Metrics
RMSE: 0.36
R² Score: -0.02
