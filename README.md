# Customer Churn Prediction using Machine Learning

## 📌 Project Overview

This project focuses on predicting **customer churn** for a telecom company using **machine learning classification models**. Customer churn refers to customers who stop using a company’s services. Accurately predicting churn helps businesses take proactive steps to retain customers.

In this project, two powerful models are implemented and compared:

* **Random Forest Classifier**
* **LightGBM (Light Gradient Boosting Machine)**

The dataset used is the popular **Telco Customer Churn Dataset**, containing customer demographics, account information, services used, and churn labels.

---

## 📂 Dataset Description

* **Rows:** 7,043 customers
* **Columns:** 21 features (after removing `customerID`)
* **Target Variable:** `Churn`

  * `1` → Customer churned
  * `0` → Customer retained

### Key Features

* Demographics: `gender`, `SeniorCitizen`, `Partner`, `Dependents`
* Services: `PhoneService`, `InternetService`, `StreamingTV`, `TechSupport`, etc.
* Account Information: `Contract`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`

---

## 🛠️ Technologies & Libraries Used

* **Python**
* **Pandas** – Data manipulation
* **NumPy** – Numerical operations
* **Matplotlib** – Data visualization
* **Scikit-learn** – Preprocessing, modeling, evaluation
* **LightGBM** – Gradient boosting model

---

## 🔄 Data Preprocessing

1. **Dropped Irrelevant Column**

   * `customerID` (not useful for prediction)

2. **Handled Data Types**

   * Converted `TotalCharges` to numeric format
   * Handled invalid values using median imputation

3. **Target Encoding**

   * `Churn`: `Yes → 1`, `No → 0`

4. **Categorical Encoding**

   * Label Encoding applied to all categorical features

5. **Feature Scaling**

   * Standardized numerical features using `StandardScaler`

6. **Train-Test Split**

   * 80% Training, 20% Testing
   * Stratified split to preserve churn distribution

---

## 📊 Exploratory Data Analysis (EDA)

The following analyses were performed:

* Churn distribution visualization
* Tenure vs Churn (Boxplot)
* Monthly Charges vs Churn (Boxplot)
* Churn rate by Contract type
* Churn rate by Payment Method

These insights help understand customer behavior and key churn drivers.

---

## 🤖 Models Implemented

### 1️⃣ Random Forest Classifier

**Configuration:**

* `n_estimators = 200`
* `class_weight = 'balanced'`
* `random_state = 42`

**Performance:**

* **Accuracy:** ~78.35%
* Better performance for non-churn customers

### 2️⃣ LightGBM Classifier

**Configuration:**

* `n_estimators = 300`
* `learning_rate = 0.05`
* `class_weight = 'balanced'`
* `random_state = 42`

**Performance:**

* **Accuracy:** ~76.36%
* Higher recall for churned customers

---

## 📈 Model Evaluation Metrics

Both models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

This provides a balanced evaluation, especially considering class imbalance.

---

## 🔍 Feature Importance Analysis

* Feature importance was extracted from the **Random Forest model**
* Top contributing features include:

  * `tenure`
  * `MonthlyCharges`
  * `TotalCharges`
  * `Contract`
  * `PaymentMethod`

A horizontal bar chart visualizes the **Top 10 most important features** influencing churn.

---

## 📌 Results Summary

| Model         | Accuracy | Churn Recall |
| ------------- | -------- | ------------ |
| Random Forest | ~78.3%   | Moderate     |
| LightGBM      | ~76.3%   | Higher       |

**Insight:**

* Random Forest performs better overall
* LightGBM captures churn customers more effectively

---

## 🚀 How to Run the Project

1. Clone the repository

```bash
git clone <repository-url>
```

2. Install dependencies

```bash
pip install pandas numpy matplotlib scikit-learn lightgbm
```

3. Run the Python script

```bash
python churn_prediction.py
```

---

## 🔮 Future Improvements

* Use **One-Hot Encoding** instead of Label Encoding
* Hyperparameter tuning using **GridSearchCV / Optuna**
* Handle class imbalance using **SMOTE**
* Add **ROC-AUC** and **Precision-Recall curves**
* Deploy the model using **Flask or FastAPI**

---

## 👤 Author

**Chakradhar Peddavenkatagari**

Aspiring AI Engineer

Masters in Computer Science

State University of New York at Buffalo
