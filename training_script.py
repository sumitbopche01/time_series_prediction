import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import joblib
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# Load data from the file
def load_data():
    logging.info("Loading data from file")
    with open("./data/nifty_5mins.json", "r") as file:
        data = json.load(file)
    logging.info("Data loaded successfully")
    return data


# Preprocess the data
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


# Generate lag features
def create_lag_features(df, lag=1):
    logging.info("Creating lag features")
    for i in range(1, lag + 1):
        df.loc[:, f"Close_lag_{i}"] = df["Close"].shift(i)
    logging.info(f"Lag features created with lag={lag}")
    return df


# Create technical indicators (e.g., moving averages)
def create_technical_indicators(df):
    logging.info("Creating technical indicators")
    df.loc[:, "MA_5"] = df["Close"].rolling(window=5).mean()
    df.loc[:, "MA_10"] = df["Close"].rolling(window=10).mean()
    logging.info("Technical indicators created")
    return df


# Perform feature engineering and train the model
def feature_engineering_and_train():
    logging.info("Starting feature engineering and model training")
    df = preprocess_data()

    # Apply feature engineering
    df = create_lag_features(df, lag=5)
    df = create_technical_indicators(df)

    # Drop NaN values created by lagging
    df.dropna(inplace=True)
    logging.info(f"DataFrame shape after dropping NaN values: {df.shape}")

    # Normalize the data
    logging.info("Normalizing the data")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(
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
    logging.info("Data normalization completed")

    # Save the scaler
    joblib.dump(scaler, "./data/scaler.joblib")
    logging.info("Scaler saved successfully")

    # Create training data
    logging.info("Creating training data")
    X_train = []
    y_train = []
    for i in range(60, len(scaled_data)):
        X_train.append(scaled_data[i - 60 : i])
        y_train.append(scaled_data[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    logging.info(
        f"Training data created with shapes: X_train={X_train.shape}, y_train={y_train.shape}"
    )

    # Build LSTM model
    logging.info("Building LSTM model")
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2])),
            tf.keras.layers.LSTM(units=50, return_sequences=True),
            tf.keras.layers.LSTM(units=50),
            tf.keras.layers.Dense(units=1),
        ]
    )

    model.compile(optimizer="adam", loss="mean_squared_error")
    logging.info("Training the model")
    model.fit(X_train, y_train, epochs=25, batch_size=32)
    logging.info("Model training completed")

    # Save the model using the native Keras format
    model.save("./data/stock_prediction_model.keras")
    logging.info("Model saved successfully in Keras format")


# Execute the feature engineering and model training function
feature_engineering_and_train()
