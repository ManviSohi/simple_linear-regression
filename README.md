
# 📊 Simple Linear Regression – Salary Prediction

## 📌 Project Overview

This project implements **Simple Linear Regression** using Python and Scikit-Learn to predict salary based on years of experience.

The project demonstrates:
- Loading a dataset
- Splitting data into training and testing sets
- Training a regression model
- Making predictions
- Visualizing results

---

## 📂 Project Structure

```
│── simple_linear_regression.py
│── Salary_Data.csv
│── README.md
```

---

## 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-Learn

---

## 📊 Dataset

The dataset contains two columns:

| Years of Experience | Salary |
|---------------------|--------|
| 1.1                 | 39343  |
| 1.3                 | 46205  |
| ...                 | ...    |

- **Independent Variable (X):** Years of Experience  
- **Dependent Variable (y):** Salary  

---

## ⚙️ Working Steps

### 1️⃣ Import Libraries

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```

### 2️⃣ Load Dataset

```python
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
```

### 3️⃣ Split Dataset

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1/3, random_state=0
)
```

### 4️⃣ Train the Model

```python
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
```

### 5️⃣ Predict Results

```python
y_pred = regressor.predict(X_test)
```

### 6️⃣ Visualize Results

- Red dots → Actual salary values  
- Blue line → Regres
