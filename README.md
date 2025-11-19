# Personal Loan Acceptance Prediction

Predict which customers are likely to accept a personal loan offer using machine learning models. This project analyzes customer demographics and financial data to help banks make data-driven decisions.

## 📌 Project Overview

The project uses a dataset similar to the Bank Marketing dataset, containing customer information such as **age, job, marital status, education, balance, housing loan, previous campaigns**, etc.
The target variable is:

* **loan_acceptance** → 1 = Accepted
* **loan_acceptance** → 0 = Not Accepted

## 🎯 Objective

* Explore customer attributes (age, job, marital status)
* Visualize loan acceptance trends
* Train **Logistic Regression** and **Decision Tree** classifiers
* Identify customer groups more likely to accept personal loans

## 📂 Dataset

Key columns in `bank.csv`:

* `age`, `job`, `marital`, `education`, `default`, `balance`, `housing`, `loan`
* `contact`, `day`, `month`, `duration`, `campaign`, `pdays`, `previous`, `poutcome`
* `deposit` → *renamed to* `loan_acceptance`

## 🧪 Data Exploration

Visualizations include:

* **Age Distribution** → `age_distribution.png`
* **Marital Status Distribution** → `marital_distribution.png`
* **Loan Acceptance by Age Group**
* **Loan Acceptance by Job**
* **Loan Acceptance by Marital Status**

> Note: These plots show general trends, while the model considers feature interactions for predictions.

## 🧠 Machine Learning Models

1. **Logistic Regression**

   * Provides probability of acceptance
   * Evaluates linear relationships

2. **Decision Tree Classifier**

   * Shows feature importance
   * Accounts for interactions between multiple features

## 📈 Evaluation

* Accuracy and classification report for both models
* Confusion matrix (`confusion_matrix_logistic.png`)
* Feature importance plot (`feature_importance.png`)

**Insights:**

* EDA shows which groups generally accept loans more often.
* Decision Tree considers combinations of features; e.g., married customers may appear more likely in certain branches due to interaction with balance, campaign, or previous contact features.

## 🛠️ How to Run

1. Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

2. Place `bank.csv` in the project folder
3. Run the Python script:

```bash
python personal_loan_prediction.py
```

4. View saved plots and evaluation metrics.
