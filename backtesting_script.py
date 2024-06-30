import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
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
        df.loc[:, f"Close_lag_{i}"] = df["Close"].shift(i)
    logging.info(f"Lag features created with lag={lag}")
    return df

def create_technical_indicators(df):
    logging.info("Creating technical indicators")
    df.loc[:, "MA_5"] = df["Close"].rolling(window=5).mean()
    df.loc[:, "MA_10"] = df["Close"].rolling(window=10).mean()
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

# Split the data into training and testing sets
train_size = int(len(df) * 0.8)
train_df, test_df = df.iloc[:train_size].copy(), df.iloc[train_size:].copy()

# Preprocess the training data
scaled_train_data = preprocess_for_prediction(train_df)

# Check if the training data has enough data
if len(scaled_train_data) <= 60:
    logging.error("Not enough data to train the model after preprocessing. Please ensure the dataset is sufficiently large.")
else:
    # Train the model
    X_train = []
    y_train = []
    for i in range(60, len(scaled_train_data)):
        X_train.append(scaled_train_data[i - 60 : i])
        y_train.append(scaled_train_data[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)

    # Check the shape of the training data
    logging.info(f"X_train shape: {X_train.shape}")
    logging.info(f"y_train shape: {y_train.shape}")

    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X_train, y_train, epochs=25, batch_size=32)
    logging.info("Model retrained on training set")

    # Preprocess the testing data
    scaled_test_data = preprocess_for_prediction(test_df)

    # Backtesting: Make predictions
    X_test = []
    y_test = test_df["Close"].values[60:]  # Get the actual close prices for comparison
    predictions = []

    for i in range(60, len(scaled_test_data)):
        X_test.append(scaled_test_data[i - 60 : i])
        X_test_array = np.array(X_test[-1]).reshape((1, 60, scaled_test_data.shape[1]))
        
        # Check the shape of the test data
        logging.info(f"X_test_array shape: {X_test_array.shape}")
        
        predicted_value = model.predict(X_test_array)
        predicted_value = scaler.inverse_transform(
            np.concatenate((predicted_value, np.zeros((predicted_value.shape[0], scaled_test_data.shape[1]-1))), axis=1)
        )[:, 0]
        predictions.append(predicted_value[0])

    # Calculate performance metrics
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    logging.info(f"Mean Squared Error: {mse}")
    logging.info(f"Mean Absolute Error: {mae}")

    # Print the results
    results = pd.DataFrame({"Actual": y_test, "Predicted": predictions})
    print(results.head(10))
    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")
