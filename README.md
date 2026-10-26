# 🚗 Uber Ride Demand Forecasting & Dynamic Pricing Optimization

An end-to-end Data Science and Business Analytics project that leverages machine learning to forecast hourly ride demand, optimize dynamic surge pricing multipliers, and quantify executive financial impact.

---

# 🚗 Uber Drives Data Analysis & Feature Engineering

An end-to-end Data Analysis and Feature Engineering project on Uber trip data, focusing on data cleaning, temporal feature extraction, time-slot categorization, and exploratory visualization of ride request patterns.

---

## 📌 Executive Summary

This project processes and analyzes a dataset of 1,155 Uber trip records. The analysis transforms raw start/end timestamps into granular temporal features, categorizes trips into distinct operational time slots, and evaluates distance distributions and hourly ride demand patterns.

### Key Dataset & Operational Insights
- **Total Records Processed:** 1,154 individual trips (excluding the total summary row).
- **Total Miles Tracked:** 12,204.7 miles across business and personal categories.
- **Primary Features Engineered:**
  - `Request Hour`, `Request Day`, `Request Date`
  - `End Hour`, `End Day`, `End Date`
  - `Duration_mins` (calculated trip duration)
  - `Time Slot` (`Early Morning`, `Morning Peak`, `Afternoon`, `Evening Peak`, `Late Night`)
  - Cyclical time features (`hour_sin`, `hour_cos`)
  - Time-series indicators & lags (`Is_Rush_Hour`, `Is_Night`, `Demand_Lag_1h`, `Demand_Lag_24h`, `Rolling_Mean_3h`)

---

## 🛠️ Tech Stack & Tools

- **Programming Language:** Python 3.x
- **Data Manipulation:** `pandas`, `numpy`
- **Data Visualization:** `matplotlib`, `seaborn`
- **Environment:** Jupyter Notebook / JupyterLab

---

## 📊 Pipeline & Analysis Architecture

1. **Data Preprocessing & Cleaning:**
   - Ingests raw trip data (`Uber Drives - .csv`).
   - Removes dataset summary/total row at index 1155.
   - Converts `START_DATE*` and `END_DATE*` columns into standard datetime objects.

2. **Feature Engineering:**
   - Extracts hour, day name, and date from start and end timestamps.
   - Computes total trip duration in minutes (`Duration_mins`).
   - Maps start hours into 5 operational time slots:
     - **Early Morning:** 04:00 – 07:59
     - **Morning Peak:** 08:00 – 11:59
     - **Afternoon:** 12:00 – 16:59
     - **Evening Peak:** 17:00 – 21:59
     - **Late Night:** 22:00 – 03:59
   - Generates indicator variables for rush hours (`08-10`, `17-19`) and night hours (`22-05`).
   - Encodes cyclical sine/cosine transformations for hours to represent 24-hour periodicity.
   - Builds lag and rolling demand features (`Demand_Lag_1h`, `Demand_Lag_24h`, `Rolling_Mean_3h`).

3. **Exploratory Data Analysis & Visualization:**
   - Analyzes distribution of ride request counts per hour of the day.
   - Maps ride frequency by trip category/purpose across time slots using cross-tabulation heatmaps.
   - Plots trip distance distributions (`MILES*`) using log-scale transformations.

---

## 📈 Visualizations Included

The notebook generates the following visual outputs:
- **Hourly Ride Request Volume:** A countplot displaying total ride requests segmented by `Request Hour`.
- **Time Slot vs. Purpose/Category Heatmap:** A cross-tabulation heatmap illustrating trip volume across time slots and business/personal purposes.
- **Trip Distance Distribution:** A histogram with KDE overlay showing trip distances (`MILES*`) on a log scale.

---

## 🚀 How to Run

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/RahulThapa-DS/Uber-Ride-Analysis.git](https://github.com/RahulThapa-DS/Uber-Ride-Analysis.git)
   cd Uber-Ride-Analysis

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
## 📋 Uber Ride Analysis — Project Deliverables Checklist

### 1. Data Pipeline & Exploratory Data Analysis
- [x] **Data Preprocessing & Cleaning:** Loaded raw ride dataset (`1,155` trips) and handled missing values.
- [x] **Feature Engineering:** Extracted temporal indicators (`Request Hour`, `Day_of_Week`, `Is_Weekend`).
- [x] **Time-Series Construction:** Built full complete hourly time-series grid (`8,784` hours).

### 2. Machine Learning & Predictive Analytics
- [x] **Model Training:** Trained a **Random Forest Regressor** to forecast hourly ride request volume.
- [x] **Feature Importance Evaluation:** Calculated and visualized feature weight rankings (`Request Hour`, `Day_of_Week`, etc.).
- [x] **Validation & Testing:** Evaluated predictive performance across the test horizon (`1,757` hours).

### 3. Dynamic Pricing & Operations Decision Engine
- [x] **Surge Multiplier Rules:** Implemented rule-based dynamic pricing based on forecasted hourly demand.
- [x] **Driver Dispatch Tiers:** Categorized fleet dispatch priority into 4 operational tiers (Tier 1 Max Bonus to Tier 4 Base Fare).
- [x] **Incentive Bonus Pool:** Allocated surge bonus payouts to optimize supply-demand equilibrium during peak hours.

### 4. Executive Dashboards & Data Visualizations
- [x] **Predictive Analytics Plot:** 2D line plot tracking hourly demand trends across days of the week.
- [x] **Operations Pie Chart:** Distribution of driver dispatch priority tiers.
- [x] **Pricing Heatmap:** Dynamic surge factor matrix (`Day_of_Week` vs. `Request Hour`).
- [x] **Financial Impact Area Chart:** Gross revenue uplift comparison (Dynamic Surge vs. Flat Base Fare).
- [x] **Interactive Dashboard:** Built consolidated 2x2 interactive Plotly subplots for executive presentation.

### 5. Business Quantification & Financial Summary
- [x] **Baseline Revenue:** Calculated flat base fare benchmark ($3,633.73).
- [x] **Optimized Gross Revenue:** Quantified dynamic surge total revenue ($6,987.92).
- [x] **Net Revenue Uplift:** Documented absolute gain (+$3,354.19 / **+92.3% Increase**).
- [x] **Driver Earnings Breakdown:** Computed platform revenue split ($1,397.58 platform fee vs. $5,590.34 driver earnings, including +$2,683.35 in surge bonuses).

---

### 6. Repository & Documentation (GitHub)
- [x] Jupyter Notebook (`Uber Ride Analysis -Project 2.ipynb`) verified and executable.
- [x] Portfolio `README.md` documentation ready with executive summary, architecture, and running instructions.
---
