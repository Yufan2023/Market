import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

def perform_eda(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)

    # Ensure categorical columns are converted to numerical values
    categorical_columns = ['Category', 'OrderPriority', 'SalesChannel', 'Country']
    label_encoder = LabelEncoder()

    for column in categorical_columns:
        if data[column].dtype == 'object':  # Only encode if the column contains strings
            data[column] = label_encoder.fit_transform(data[column])

    # Convert 'Discount' to numeric, replacing non-numeric values with 0
    data['Discount'] = pd.to_numeric(data['Discount'], errors='coerce').fillna(0)

    # Calculate 'TotalSpent' if not already present
    data['TotalSpent'] = data['Quantity'] * data['UnitPrice']

    # Feature Engineering
    data['Discount_Category'] = data['Discount'] * data['Category']
    data['Discount_SalesChannel'] = data['Discount'] * data['SalesChannel']
    data['Discount_OrderPriority'] = data['Discount'] * data['OrderPriority']

    # Plot Correlation Matrix
    features_to_analyze = [
        'Quantity', 'Discount', 'TotalSpent', 'Category', 'SalesChannel',
        'OrderPriority', 'Discount_Category', 'Discount_SalesChannel', 'Discount_OrderPriority'
    ]
    correlation_matrix = data[features_to_analyze].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', square=True)
    plt.title('Feature Correlation Matrix')
    plt.show()

    # Distribution Plots
    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    sns.histplot(data['Discount'], bins=20, kde=True)
    plt.title('Discount Distribution')

    plt.subplot(1, 3, 2)
    sns.histplot(data['Quantity'], bins=20, kde=True)
    plt.title('Quantity Distribution')

    plt.subplot(1, 3, 3)
    sns.histplot(data['TotalSpent'], bins=20, kde=True)
    plt.title('TotalSpent Distribution')

    plt.show()

    # Scatter Plot
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=data, x='Discount', y='Quantity', hue='Category', palette='viridis')
    plt.title('Discount vs Quantity by Category')
    plt.xlabel('Discount')
    plt.ylabel('Quantity')
    plt.legend(title='Category')
    plt.show()

    # Box Plots
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.boxplot(data=data, x='SalesChannel', y='Quantity')
    plt.title('Quantity by SalesChannel')

    plt.subplot(1, 2, 2)
    sns.boxplot(data=data, x='SalesChannel', y='TotalSpent')
    plt.title('TotalSpent by SalesChannel')

    plt.tight_layout()
    plt.show()

    # Heatmap of Average TotalSpent by Category and OrderPriority
    category_order_spent = data.pivot_table(values='TotalSpent', index='Category', columns='OrderPriority', aggfunc='mean')

    plt.figure(figsize=(10, 6))
    sns.heatmap(category_order_spent, annot=True, cmap='YlGnBu')
    plt.title('Average TotalSpent by Category and OrderPriority')
    plt.xlabel('Order Priority')
    plt.ylabel('Category')
    plt.show()

    # Box Plot by Discount Range
    bins = [0, 0.1, 0.3, 0.5, 1]
    labels = ['No Discount', 'Low', 'Medium', 'High']
    data['DiscountRange'] = pd.cut(data['Discount'], bins=bins, labels=labels)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    sns.boxplot(data=data, x='DiscountRange', y='Quantity', palette='Set3')
    plt.title('Quantity by Discount Range')

    plt.subplot(1, 2, 2)
    sns.boxplot(data=data, x='DiscountRange', y='TotalSpent', palette='Set3')
    plt.title('TotalSpent by Discount Range')

    plt.tight_layout()
    plt.show()
