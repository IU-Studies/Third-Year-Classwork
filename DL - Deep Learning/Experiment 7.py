

"""
Develop RNN/LSTM/GRU model for suitable application. 
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    SimpleRNN,
    LSTM,
    GRU
)

import yfinance as yf

# ==============================
# DOWNLOAD STOCK DATA
# ==============================

data = yf.download(
    "AAPL",
    start="2015-01-01",
    end="2023-01-01"
)

# Use closing prices
dataset = data['Close'].values.reshape(-1, 1)

# ==============================
# NORMALIZE DATA
# ==============================

scaler = MinMaxScaler(feature_range=(0, 1))

dataset_scaled = scaler.fit_transform(dataset)

# ==============================
# CREATE DATASET FUNCTION
# ==============================

def create_dataset(data, time_step=50):

    X, Y = [], []

    for i in range(len(data) - time_step - 1):

        X.append(data[i:(i + time_step), 0])

        Y.append(data[i + time_step, 0])

    return np.array(X), np.array(Y)

# ==============================
# PREPARE INPUT DATA
# ==============================

time_step = 50

X, y = create_dataset(dataset_scaled, time_step)

# Reshape input for RNN/LSTM/GRU
X = X.reshape(X.shape[0], X.shape[1], 1)

# ==============================
# TRAIN-TEST SPLIT
# ==============================

split = int(len(X) * 0.8)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ==============================
# BUILD RNN MODEL
# ==============================

def build_rnn():

    model = Sequential()

    model.add(
        SimpleRNN(
            50,
            input_shape=(time_step, 1)
        )
    )

    model.add(Dense(1))

    model.compile(
        optimizer='adam',
        loss='mse'
    )

    return model

# ==============================
# BUILD LSTM MODEL
# ==============================

def build_lstm():

    model = Sequential()

    model.add(
        LSTM(
            50,
            input_shape=(time_step, 1)
        )
    )

    model.add(Dense(1))

    model.compile(
        optimizer='adam',
        loss='mse'
    )

    return model

# ==============================
# BUILD GRU MODEL
# ==============================

def build_gru():

    model = Sequential()

    model.add(
        GRU(
            50,
            input_shape=(time_step, 1)
        )
    )

    model.add(Dense(1))

    model.compile(
        optimizer='adam',
        loss='mse'
    )

    return model

# ==============================
# CREATE MODELS
# ==============================

models = {
    "RNN": build_rnn(),
    "LSTM": build_lstm(),
    "GRU": build_gru()
}

predictions = {}

EPOCHS = 5

# ==============================
# TRAIN MODELS
# ==============================

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=32,
        verbose=1
    )

    pred = model.predict(X_test)

    predictions[name] = scaler.inverse_transform(pred)

# ==============================
# PLOT RESULTS
# ==============================

plt.figure(figsize=(10, 6))

# Actual prices
plt.plot(
    scaler.inverse_transform(
        y_test.reshape(-1, 1)
    ),
    label="Actual",
    color='black'
)

# Predicted prices
for name, pred in predictions.items():

    plt.plot(
        pred,
        label=name
    )

plt.title("RNN vs LSTM vs GRU (Stock Prediction)")

plt.xlabel("Time")
plt.ylabel("Stock Price")

plt.legend()

plt.show()
