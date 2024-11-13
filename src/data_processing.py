import pandas as pd

def load_and_process_data(file_path):
    data = pd.read_csv(file_path)
    data['TotalSpent'] = data['Quantity'] * data['UnitPrice']
    data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
    clv_data = data.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (x.max() - x.min()).days,
        'InvoiceNo': 'nunique',
        'TotalSpent': 'sum'
    }).rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency'})
    clv_data['AvgOrderValue'] = clv_data['TotalSpent'] / clv_data['Frequency']
    clv_data.fillna(0, inplace=True)
    return clv_data
