import pandas as pd
import ta

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ── Moving Averages ───────────────────────────────────
    df["sma_20"] = ta.trend.sma_indicator(df["Close"], window=20)
    df["sma_50"] = ta.trend.sma_indicator(df["Close"], window=50)
    df["price_above_sma20"] = (df["Close"] > df["sma_20"]).astype(int)
    df["price_above_sma50"] = (df["Close"] > df["sma_50"]).astype(int)

    # ── RSI ───────────────────────────────────────────────
    df["rsi"] = ta.momentum.rsi(df["Close"], window=14)
    df["rsi_overbought"] = (df["rsi"] > 70).astype(int)
    df["rsi_oversold"] = (df["rsi"] < 30).astype(int)

    # ── MACD ──────────────────────────────────────────────
    macd = ta.trend.MACD(df["Close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_histogram"] = macd.macd_diff()
    df["macd_bullish"] = (df["macd"] > df["macd_signal"]).astype(int)

    # ── Bollinger Bands ───────────────────────────────────
    bollinger = ta.volatility.BollingerBands(df["Close"], window=20)
    df["bb_upper"] = bollinger.bollinger_hband()
    df["bb_lower"] = bollinger.bollinger_lband()
    df["bb_width"] = df["bb_upper"] - df["bb_lower"]
    df["bb_position"] = (df["Close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    # ── Volume ────────────────────────────────────────────
    df["volume_sma20"] = ta.trend.sma_indicator(df["Volume"], window=20)
    df["volume_ratio"] = df["Volume"] / df["volume_sma20"]
    df["obv"] = ta.volume.on_balance_volume(df["Close"], df["Volume"])

    # ── Lag features ──────────────────────────────────────
    for lag in [1, 2, 3, 5]:
        df[f"return_lag{lag}"] = df["Close"].pct_change(lag)
        df[f"volume_lag{lag}"] = df["Volume"].pct_change(lag)

    # ── Target variable ───────────────────────────────────
    # New — predicts significant moves only
    next_return = df["Close"].pct_change(-1)
    df["target"] = None
    df.loc[next_return > 0.01, "target"] = 1   # UP > 1%
    df.loc[next_return < -0.01, "target"] = 0  # DOWN > 1%
    df = df.dropna(subset=["target"])
    df["target"] = df["target"].astype(int)

    # ── Drop NaN rows ─────────────────────────────────────
    df = df.dropna()

    print(f"Features created: {len(df.columns)} columns")
    print(f"Trading days after cleaning: {len(df)}")
    print(f"Target distribution: {df['target'].value_counts().to_dict()}")

    return df