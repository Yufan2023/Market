import pandas as pd

def predict_clv(model, recency, frequency, avg_order_value):
    new_data = pd.DataFrame({
        'Recency': [recency],
        'Frequency': [frequency],
        'AvgOrderValue': [avg_order_value]
    })
    return model.predict(new_data)[0]
