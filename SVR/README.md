# 🍽️ Support Vector Regression on Tips Dataset

This project implements a **Support Vector Regression (SVR)** model using the popular **Tips dataset** from Seaborn. The goal is to predict the `total_bill` a customer pays at a restaurant based on various features like tip amount, day, time, gender, smoker status, and party size.

---

## 📊 Dataset

- **Source**: `seaborn.load_dataset('tips')`
- **Target Variable**: `total_bill`
- **Features Used**:
  - `tip`
  - `sex`
  - `smoker`
  - `day`
  - `time`
  - `size`

---

## 🛠️ Preprocessing Steps

- **Label Encoding**:
  - Applied to `sex`, `smoker`, and `time` columns using `LabelEncoder`.
- **One-Hot Encoding**:
  - Applied to `day` column using `ColumnTransformer` and `OneHotEncoder(drop='first')`.
- **Train-Test Split**:
  - 75% Training / 25% Testing

---

## 🧠 Model

- **Algorithm**: `Support Vector Regression` (from `sklearn.svm`)
- **Initial Training**: Fit on transformed `X_train`, predict on `X_test`
- **Evaluation Metrics**:
  - R² Score
  - Mean Absolute Error (MAE)

---

## 🔍 Hyperparameter Tuning

Used `GridSearchCV` to find optimal hyperparameters:
```python
param_grid = {
  'C': [0.1, 1, 10, 100, 1000],
  'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
  'kernel': ['rbf']
}
