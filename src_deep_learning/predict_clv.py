import pandas as pd
from model_definition import build_model

def load_trained_model(input_dim, weights_path='saved_model_weights.h5'):
    model = build_model(input_dim)
    model.load_weights(weights_path)
    return model

def prepare_input_data(new_data, scaler):
    new_data_df = pd.DataFrame(new_data)
    scaled_data = scaler.transform(new_data_df)
    return scaled_data

def predict_clv(model, scaled_data):
    return model.predict(scaled_data)[0][0]
