"""
4_deep_learning_forecasting/gru_model.py
GRU forecasting model (parallel to LSTM pipeline).
"""

import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from utils.preprocess import load_data
from utils.common import ensure_outdir

OUT_DIR = os.path.join("4_deep_learning_forecasting", "outputs")
ensure_outdir(OUT_DIR)

SEQ_LEN = 14
PRED_HORIZON = 7
FEATURES = ["stringency_index", "cases_7d", "log_growth", "population"]


def build_sequences(df_country):
    arr = df_country[FEATURES].fillna(method="ffill").fillna(0).values
    y = df_country["cases_7d"].fillna(0).values

    Xs, ys = [], []
    for i in range(len(arr) - SEQ_LEN - PRED_HORIZON):
        Xs.append(arr[i:i + SEQ_LEN])
        ys.append(np.mean(y[i + SEQ_LEN:i + SEQ_LEN + PRED_HORIZON]))
    return np.array(Xs), np.array(ys)


def train_gru(country, df):
    dfc = df[df["country"].str.lower() == country.lower()].sort_values("date")

    if len(dfc) < SEQ_LEN + PRED_HORIZON + 20:
        return None

    X, y = build_sequences(dfc)

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train.reshape(-1, X.shape[2])).reshape(X_train.shape)
    X_test_s = scaler.transform(X_test.reshape(-1, X.shape[2])).reshape(X_test.shape)

    y_scaler = MinMaxScaler()
    y_train_s = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()

    model = Sequential([
        GRU(64, return_sequences=True, input_shape=(SEQ_LEN, X.shape[2])),
        GRU(32),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")

    model.fit(X_train_s, y_train_s, validation_split=0.1,
              epochs=30, batch_size=32, verbose=1,
              callbacks=[EarlyStopping(patience=5)])

    preds = model.predict(X_test_s).ravel()
    preds_inv = y_scaler.inverse_transform(preds.reshape(-1, 1)).ravel()

    out_path = os.path.join(OUT_DIR, f"gru_preds_{country}.csv")
    pd.DataFrame({"y_true": y_test, "y_pred": preds_inv}).to_csv(out_path, index=False)

    print("Saved GRU predictions for", country)
    return model


if __name__ == "__main__":
    df = load_data()
    df.columns = df.columns.str.lower()
    for c in ["India", "Japan", "Brazil"]:
        train_gru(c, df)
