# main.py

from src.data_processing import load_and_process_data
from src.eda_visualizations import perform_eda
from src.model_training import train_model
from src.predict_clv_regression import predict_clv

def main():
    # File path for the dataset
    file_path = "/data/online_sales_dataset.csv"  # Update with your dataset path if different

    # Step 1: Perform EDA
    print("Performing Exploratory Data Analysis...")
    perform_eda(file_path)

    # Step 2: Load and Process Data
    print("Loading and processing data...")
    clv_data = load_and_process_data(file_path)
    print("Data loaded and processed successfully.")
    
    # Step 3: Train the Model
    print("Training the model...")
    model = train_model(clv_data)
    print("Model trained successfully.")
    
    # Step 4: Make Predictions
    print("Making predictions...")
    recency = 30  # Example Recency value in days
    frequency = 5  # Example Frequency value
    avg_order_value = 200  # Example Average Order Value in currency units
    predicted_clv = predict_clv(model, recency, frequency, avg_order_value)
    print(f"Predicted CLV for Recency={recency}, Frequency={frequency}, AvgOrderValue={avg_order_value}: {predicted_clv}")

if __name__ == "__main__":
    main()
