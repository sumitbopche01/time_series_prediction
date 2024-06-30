import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the model
model = tf.keras.models.load_model('./data/stock_prediction_model.keras')
logging.info("Model loaded successfully")

# Load the scaler
scaler = joblib.load('./data/scaler.joblib')
logging.info("Scaler loaded successfully")

# Load and preprocess the data
def load_data():
    logging.info("Loading data from file")
    with open("./data/nifty_5mins.json", "r") as file:
        data = json.load(file)
    logging.info("Data loaded successfully")
    return data

def preprocess_data():
    logging.info("Preprocessing data")
    data = load_data()
    records = data["data"]["candles"]
    df = pd.DataFrame(
        records,
        columns=["Timestamp", "Open", "High", "Low", "Close", "Volume", "Extra"],
    )
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    logging.info(f"DataFrame shape after preprocessing: {df.shape}")
    logging.debug(f"DataFrame head:\n{df.head()}")
    return df

def create_lag_features(df, lag=1):
    logging.info("Creating lag features")
    for i in range(1, lag + 1):
        df[f"Close_lag_{i}"] = df["Close"].shift(i)
    logging.info(f"Lag features created with lag={lag}")
    return df

def create_technical_indicators(df):
    logging.info("Creating technical indicators")
    df["MA_5"] = df["Close"].rolling(window=5).mean()
    df["MA_10"] = df["Close"].rolling(window=10).mean()
    logging.info("Technical indicators created")
    return df

def preprocess_for_prediction(df):
    # Apply the same feature engineering as during training
    df = create_lag_features(df, lag=5)
    df = create_technical_indicators(df)
    
    # Drop NaN values created by lagging
    df.dropna(inplace=True)
    
    # Normalize the data
    scaled_data = scaler.transform(
        df[
            [
                "Close",
                "Close_lag_1",
                "Close_lag_2",
                "Close_lag_3",
                "Close_lag_4",
                "Close_lag_5",
                "MA_5",
                "MA_10",
            ]
        ]
    )
    
    return scaled_data

# Preprocess the data
df = preprocess_data()
scaled_data = preprocess_for_prediction(df)

# Get the last 60 values for prediction
X_input = scaled_data[-60:]
X_input = np.array(X_input).reshape((1, X_input.shape[0], X_input.shape[1]))

# Make the prediction
predicted_value = model.predict(X_input)

# Inverse transform the prediction to get the original scale
predicted_value = scaler.inverse_transform(np.concatenate((predicted_value, np.zeros((predicted_value.shape[0], scaled_data.shape[1]-1))), axis=1))[:, 0]

# Get the next timestamp
last_timestamp = df['Timestamp'].iloc[-1]
next_timestamp = last_timestamp + pd.Timedelta(minutes=5)  # Assuming the data is in 5-minute intervals

print(f"Predicted Close Price for {next_timestamp}: {predicted_value[0]}")
