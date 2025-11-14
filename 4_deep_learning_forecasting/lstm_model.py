"""
4_deep_learning_forecasting/lstm_model.py
LSTM model for forecasting 7-day-ahead COVID-19 case trends.
"""

import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from utils.preprocess import load_data
from utils.common import ensure_outdir

OUT_DIR = os.path.join("4_deep_learning_forecasting", "outputs")
ensure_outdir(OUT_DIR)

SEQ_LEN = 14
PRED_HORIZON = 7
FEATURES = ["stringency_index", "cases_7d", "log_growth", "population"]
EPOCHS = 40
BATCH = 32


def build_sequences(df_country):
    """Create sliding 14-day windows → predict next 7 days avg."""
    arr = df_country[FEATURES].fillna(method="ffill").fillna(0).values
    y = df_country["cases_7d"].fillna(0).values

    Xs, ys = [], []
    for i in range(len(arr) - SEQ_LEN - PRED_HORIZON):
        Xs.append(arr[i:i + SEQ_LEN])
        ys.append(np.mean(y[i + SEQ_LEN:i + SEQ_LEN + PRED_HORIZON]))
    return np.array(Xs), np.array(ys)


def train_lstm(country, df):
    dfc = df[df["country"].str.lower() == country.lower()].sort_values("date")
    if len(dfc) < SEQ_LEN + PRED_HORIZON + 20:
        print("Not enough data:", country)
        return

    X, y = build_sequences(dfc)

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Scale features
    feature_scaler = MinMaxScaler()
    X_train_s = feature_scaler.fit_transform(X_train.reshape(-1, X.shape[2])).reshape(X_train.shape)
    X_test_s = feature_scaler.transform(X_test.reshape(-1, X.shape[2])).reshape(X_test.shape)

    # Scale target
    y_scaler = MinMaxScaler()
    y_train_s = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()

    # Build model
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, X.shape[2])),
        Dropout(0.2),
        LSTM(32),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")

    # Callbacks
    save_path = os.path.join(OUT_DIR, f"lstm_{country.replace(' ', '_')}.h5")
    callbacks = [
        EarlyStopping(patience=6, restore_best_weights=True),
        ModelCheckpoint(save_path, save_best_only=True)
    ]

    # Train
    model.fit(X_train_s, y_train_s, validation_split=0.1,
              epochs=EPOCHS, batch_size=BATCH, verbose=1, callbacks=callbacks)

    # Predictions
    preds = model.predict(X_test_s).ravel()
    preds_inv = y_scaler.inverse_transform(preds.reshape(-1, 1)).ravel()

    out = pd.DataFrame({"country": country, "y_true": y_test, "y_pred": preds_inv})
    out.to_csv(os.path.join(OUT_DIR, f"lstm_preds_{country}.csv"), index=False)

    print(f"LSTM forecasting completed for {country}")
    return model


if __name__ == "__main__":
    df = load_data()
    df.columns = df.columns.str.lower()

    for country in ["India", "United States", "United Kingdom"]:
        train_lstm(country, df)
