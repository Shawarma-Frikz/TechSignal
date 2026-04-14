import yfinance as yf
import pandas as pd

def load_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start_date, end=end_date)
    df.columns = df.columns.droplevel(1)
    df = df.dropna()

    print(f"Loaded {len(df)} trading days for {ticker}")
    print(f"From {df.index[0].date()} to {df.index[-1].date()}")
    print(df.tail())

    return df
