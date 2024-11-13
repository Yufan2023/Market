from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def train_model(clv_data, profit_margin=0.05):
    clv_data['CLV'] = clv_data['AvgOrderValue'] * clv_data['Frequency'] * profit_margin
    X = clv_data[['Recency', 'Frequency', 'AvgOrderValue']]
    y = clv_data['CLV']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("Mean Squared Error:", mse)
    print("R-squared:", r2)
    return model
