# 🚗 Uber Ride Demand Forecasting & Dynamic Pricing Optimization

An end-to-end Data Science and Business Analytics project that leverages machine learning to forecast hourly ride demand, optimize dynamic surge pricing multipliers, and quantify executive financial impact.

---

## 📌 Executive Summary

Modern ride-hailing platforms balance driver availability and passenger demand through dynamic pricing. This project builds a machine learning pipeline using a **Random Forest Regressor** to predict hourly ride request volumes. Based on forecasted demand, the engine dynamically adjusts surge multipliers to:
- Maximize revenue during peak demand windows.
- Allocate driver dispatch priorities to high-demand clusters.
- Balance supply and demand through structured driver surge incentive bonuses.

### Key Financial & Operational Results
- **Raw Trips Analyzed:** 1,155 rides across an 8,784-hour time-series grid.
- **Baseline Gross Revenue (Flat Fare):** $3,633.73
- **Dynamic Optimized Revenue:** $6,987.92
- **Net Revenue Uplift:** **+$3,354.19 (+92.3% Increase)**
- **Driver Earnings (80% Payout):** $5,590.34 (includes +$2,683.35 in surge incentive bonuses)

---

## 🛠️ Tech Stack & Tools

- **Programming Language:** Python 3.x
- **Data Manipulation:** `pandas`, `numpy`
- **Machine Learning:** `scikit-learn` (Random Forest Regressor)
- **Data Visualization:** `matplotlib`, `seaborn`, `plotly`
- **Environment:** Jupyter Notebook / JupyterLab

---

## 📊 Pipeline & Decision Engine Architecture

1. **Feature Engineering & Time-Series Modeling:**
   - Extracts temporal features including `Request Hour`, `Day_of_Week`, and `Is_Weekend`.
   - Constructs a complete hourly time-series grid to accurately model demand variations.

2. **Predictive Analytics:**
   - Random Forest Regressor models demand trends across different days and peak hours.

3. **Dynamic Pricing & Dispatch Logic:**
   - Assigns dynamic surge multipliers based on forecasted ride volume.
   - Categorizes driver dispatch priority into operational tiers (Tier 1 Max Bonus, Tier 2 Medium, Tier 3 Standard, Tier 4 Base Fare).

4. **Financial Quantification:**
   - Computes platform take-rate revenue (20% Uber platform fee) vs. driver payout (80%).
   - Evaluates surge revenue uplift against standard flat base fare models.

---

## 📈 Visualizations & Dashboards

The repository includes scripts to generate interactive and static executive dashboards:
- **Predictive Analytics:** Hourly demand trendlines segmented by day of the week.
- **Operations:** Driver dispatch allocation priority pie distribution.
- **Pricing Engine:** Hourly surge factor heatmap (`Day_of_Week` vs. `Request Hour`).
- **Financial Quantification:** Area chart displaying net revenue uplift over flat base fare.

---

## 🚀 How to Run

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/RahulThapa-DS/Uber-Ride-Analysis.git](https://github.com/RahulThapa-DS/Uber-Ride-Analysis.git)
   cd Uber-Ride-Analysis
   ```
2. **Install required packages:**

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```
3. **Launch the notebook:**

```bash
jupyter notebook "Uber Ride Analysis -Project 2.ipynb"
```
---
## 📋 Project Deliverables Checklist
* **Python Jupyter Notebook (.ipynb) codebase**

* **Data Preprocessing & Clean Pipeline**

* **Seaborn & Matplotlib Visualization Suite**

* **Predictive Model & Metrics Evaluation**

* **Documented README Summary Report**
---
