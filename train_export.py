# train_export.py
import os
import numpy as np
import pandas as pd
import joblib

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import MinMaxScaler

def load_data(csv_path):
    df = pd.read_csv(csv_path, parse_dates=True, index_col=0, infer_datetime_format=True)
    if 'Close' not in df.columns:
        raise ValueError("CSV must contain 'Close' column")
    return df

def prepare_series(df, feature='Close', timestep=60, train_fraction=0.8):
    dataset = df[[feature]]
    data = dataset.values
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled = scaler.fit_transform(data.reshape(-1,1))

    train_size = int(len(scaled) * train_fraction)
    train_data = scaled[:train_size]
    
    # prepare x_train, y_train
    x_train, y_train = [], []
    for i in range(timestep, len(train_data)):
        x_train.append(train_data[i-timestep:i, 0])
        y_train.append(train_data[i, 0])
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    return x_train, y_train, scaler

def build_model(input_timesteps):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(input_timesteps, 1)),
        LSTM(64, return_sequences=False),
        Dense(32),
        Dense(16),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train(csv_path, model_out="models/gold_lstm_model.h5", scaler_out="models/scaler.save", epochs=50, batch_size=32):
    # This block defines 'model', 'x_train', and 'scaler'
    df = load_data(csv_path)
    timestep = 60
    x_train, y_train, scaler = prepare_series(df, timestep=timestep)

    # *** CRUCIAL: 'model' is defined here by calling build_model ***
    model = build_model(x_train.shape[1])
    
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)]
    
    print("Starting model training...")
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=2)

    # Save model and scaler
    # NOTE: Added include_optimizer=False fix for Keras version compatibility
    model.save(model_out, save_format='h5', include_optimizer=False)
    joblib.dump(scaler, scaler_out)
    print(f"\nSaved model to {model_out} and scaler to {scaler_out}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # Correct paths for local execution
    parser.add_argument("--csv", type=str, default="data/goldstock.csv", help="path to CSV with Close column")
    parser.add_argument("--model-out", type=str, default="models/gold_lstm_model.h5")
    parser.add_argument("--scaler-out", type=str, default="models/scaler.save")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    
    # Ensure directories exist before saving
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    os.makedirs(os.path.dirname(args.scaler_out), exist_ok=True)
    
    try:
        # The script will now crash here if the files are not found, before the 'model' is defined error.
        train(args.csv, args.model_out, args.scaler_out, epochs=args.epochs, batch_size=args.batch_size)
    except Exception as e:
        print(f"Error during training: {e}")
        print("Please ensure 'data/goldstock.csv' exists and contains a 'Close' column.")