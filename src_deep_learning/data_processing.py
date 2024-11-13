import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
    data['TotalSpent'] = data['Quantity'] * data['UnitPrice']
    reference_date = data['InvoiceDate'].max() + pd.Timedelta(days=1)

    clv_data = data.groupby('CustomerID').agg({
        'InvoiceDate': [
            lambda x: (reference_date - x.max()).days,
            lambda x: (reference_date - x.min()).days,
        ],
        'InvoiceNo': 'nunique',
        'TotalSpent': 'sum'
    }).reset_index()

    clv_data.columns = ['CustomerID', 'Recency', 'CustomerAge', 'Frequency', 'Monetary']
    clv_data['CLV'] = clv_data['Monetary']

    X = clv_data[['Recency', 'CustomerAge', 'Frequency', 'Monetary']]
    y = clv_data['CLV']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test
