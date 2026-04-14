import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = [
    "sma_20", "sma_50",
    "price_above_sma20", "price_above_sma50",
    "rsi", "rsi_overbought", "rsi_oversold",
    "macd", "macd_signal", "macd_histogram", "macd_bullish",
    "bb_width", "bb_position",
    "volume_ratio", "obv",
    "return_lag1", "return_lag2", "return_lag3", "return_lag5",
    "volume_lag1", "volume_lag2", "volume_lag3", "volume_lag5",
]

def split_and_scale(df: pd.DataFrame):
    # Chronological split — 80% train, 20% test
    split_idx = int(len(df) * 0.8)
    train = df.iloc[:split_idx]
    test  = df.iloc[split_idx:]

    print(f"Training: {len(train)} days ({train.index[0].date()} to {train.index[-1].date()})")
    print(f"Testing:  {len(test)} days ({test.index[0].date()} to {test.index[-1].date()})")
    print(f"Up days in train: {train['target'].sum()} / {len(train)}")
    print(f"Up days in test:  {test['target'].sum()} / {len(test)}")

    # Separate features and target
    X_train = train[FEATURE_COLS]
    y_train = train["target"]
    X_test  = test[FEATURE_COLS]
    y_test  = test["target"]

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler