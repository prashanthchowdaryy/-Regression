#  Machine Learning Models — From Scratch to Mastery

This repository contains my daily Machine Learning practice projects, covering core algorithms in Regression and Classification.

Each model is implemented using Python and scikit-learn with clear visualization and explanations.

##  Models Covered

### 🔹 Regression
- Linear Regression
- Polynomial Regression
- Support Vector Regression (SVR)
- Decision Tree Regression
- Random Forest Regression

### 🔹 Classification
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Naive Bayes
- Decision Tree Classifier

## 🛠 Tech Stack
- Python
- NumPy
- Pandas
- Matplotlib
- scikit-learn

##  Goal
To build strong intuition behind ML algorithms through consistent hands-on implementation.

---

Daily progress. One model at a time 🚀

# Polynomial Regression

Polynomial Regression is an extension of Linear Regression used to model **non-linear relationships** between the independent and dependent variables by transforming input features into polynomial features.

It keeps the simplicity of Linear Regression while adding the power to fit curves.

---

##  Problem Statement
Predict **salary** based on **position level**, where the relationship is non-linear and cannot be accurately captured by a straight line.

---

##  Model Explanation
- Uses `PolynomialFeatures` to expand input features into higher-degree terms
- Applies `LinearRegression` on the transformed features
- Degree of polynomial controls model complexity

> Polynomial Regression = Linear Regression + Feature Transformation

---

##  Tech Stack
- Python  
- NumPy  
- Matplotlib  
- scikit-learn  

---

##  Workflow
1. Load the dataset
2. Transform features using Polynomial Features
3. Train Linear Regression on transformed data
4. Visualize polynomial curve
5. Predict salary for a given position level

---

##  Visualization
- **Red dots** → Actual data points  
- **Blue curve** → Polynomial regression prediction  

---

##  Sample Prediction
```python
poly_reg.predict(poly.fit_transform([[6.5]]))
