# Customer Analytics and Lifetime Value Prediction

This repository contains a comprehensive project for customer analytics, regional shopping behavior analysis, and Customer Lifetime Value (CLV) prediction. It includes both traditional machine learning and deep learning approaches for analyzing customer data and predicting key business metrics.

---

## Features

1. **Customer Lifetime Value Prediction (Regression)**:
   - **Machine Learning**: Predict CLV using Random Forest Regression.
   - **Deep Learning**: Predict CLV using a fully connected neural network built with TensorFlow and Keras.
   
2. **Regional Shopping Behavior Analysis**:
   - Analyze and cluster regions based on shopping behaviors using K-Means clustering and PCA for dimensionality reduction.

3. **Exploratory Data Analysis (EDA)**:
   - Generate visualizations and analyze feature correlations to understand data patterns.

4. **Custom Feature Engineering**:
   - Perform advanced feature engineering for Discount, Category, and Sales Channel analysis.

---


## Dataset

The dataset (`online_sales_dataset.csv`) contains the following features:
- **CustomerID**: Unique identifier for each customer.
- **InvoiceNo**: Unique identifier for each invoice.
- **InvoiceDate**: Date of the invoice.
- **Quantity**: Quantity of products purchased.
- **UnitPrice**: Price per unit.
- **Discount**: Discount applied to the purchase.
- **Category**: Product category.
- **SalesChannel**: Sales channel type (e.g., online or offline).
- **Country**: Customer's region or country.
- **ReturnStatus**: Indicates if a product was returned.

---

## Getting Started

### Prerequisites

- Python 3.8+
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
## How to Use

### 1. Run EDA
Analyze the dataset and generate visualizations:
```bash
python src/eda_visualizations.py
```

### 2. Run Regression Model for CLV Prediction
Train and predict CLV using the regression-based approach:
```bash
python main_for_regression.py
```

### 3. Run Deep Learning Model for CLV Prediction
Train and predict CLV using the deep learning pipeline:
```bash
python main_deep_learning.py
```

### 4. Analyze Regional Shopping Behavior
Cluster and visualize shopping behavior across regions:
```bash
python Regional Shopping Behavior.py
```

## Visualizations

 - Feature correlations for regression and deep learning models.
 - Predictions of CLV based on Recency, Frequency, and Monetary features.
 - Clustering regions based on total spending, return rates, and product preferences.
 - PCA-based visualization of regional clusters.

## Dependencies
- ```pandas```
- ```matplotlib```
- ```seaborn```
- ```scikit-learn```
- ```tensorflow```
