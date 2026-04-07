 Manufacturing Defect Prediction
## Using Logistic Regression on Unified Production Data

> **Ivy Professional School | Data Science Internship 2026**
> Intern: Dhara Nayak

---

## Project Overview

Manufacturing plants lose significant revenue due to **undetected product defects**
that reach downstream processes or end customers. This project builds a machine
learning model that predicts whether a production batch will be defective —
**before it leaves the production line**.

The model was trained on **12,203 real production records** integrated from
**6 operational systems** (ERP, MES, QMS, SCADA, Machine, Operator).

---

## Model Results

| Metric | Score |
|--------|-------|
| Accuracy | **93.6%** |
| Precision | **95.7%** |
| Recall (Defect) | **91.2%** |
| F1-Score | **93.4%** |
| ROC-AUC | **98.8%** |

> Recall is prioritised — in manufacturing, **missing a defect costs more than a false alarm**.

---

## Dataset

| Property | Value |
|----------|-------|
| Total Records | 12,203 production batches |
| Features | 38 columns |
| Data Sources | 6 systems (ERP, MES, QMS, SCADA, Machine, Operator) |
| Target Variable | defect_flag (0 = Normal, 1 = Defect) |
| Class Balance | ~50/50 balanced |

---

## Project Structure

```
manufacturing-defect-prediction/
├── Dhara_Nayak_Task4_Final_V2.ipynb  # Main analysis notebook
├── Dhara_Nayak_Final_PPT.pdf         # Presentation slides
├── requirements.txt                   # Python dependencies
├── data/                             # Dataset (if included)
└── images/                           # Charts and visualisations
```

---

## Key Findings

- **rework_min** is the strongest predictor of defects (r = 0.573)
- Classes are naturally balanced: ~50% Normal, ~50% Defect
- IQR-based outlier capping preserved all 12,203 rows (no data deleted)
- GridSearchCV with 5-fold CV selected optimal hyperparameters

---

## Methodology

### 1. Data Preparation
- Merged 6 source systems into one unified dataset
- Missing values: Numerical → Median | Categorical → Mode
- Outlier treatment: IQR Winsorization (capping, no rows deleted)

### 2. Exploratory Data Analysis
- Univariate: Histograms, KDE plots, Box plots for all 38 features
- Bivariate: Scatter plots and category box plots vs scrap_pct
- Correlation: Pearson r values for all numerical features

### 3. Feature Engineering
- Created binary target: defect_flag (scrap_pct > median = 1)
- Dropped ID columns, date columns, and leakage features
- One-Hot Encoding for all categorical features (dtype=int)
- StandardScaler applied on train set only (no test leakage)

### 4. Model Building
- Baseline: Logistic Regression (lbfgs, max_iter=1000)
- Improved: GridSearchCV with StratifiedKFold (5-fold)
  - C: [0.1, 1, 10]
  - Penalty: [l1, l2]
  - class_weight: [None, balanced]
- 5-Fold Cross-Validation for stability check

---

## How to Run

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/manufacturing-defect-prediction.git
cd manufacturing-defect-prediction

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook Dhara_Nayak_Task4_Final_V2.ipynb
```

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2+-orange)
![Pandas](https://img.shields.io/badge/Pandas-1.5+-green)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)

---

## Business Impact

The model enables **proactive quality control**:
- Flag high-risk batches before completion
- Alert supervisors when defect probability > 50%
- Focus process improvement on rework_min (top defect driver)
- Estimated reduction in inspection cost and scrap waste

---

*Ivy Professional School | Data Science Internship 2026 | Dhara Nayak*
