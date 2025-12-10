import yfinance as yf
import pandas as pd
import os
from datetime import datetime

# 1. Setup Path
folder_path = r"C:\Users\hunne\OneDrive - University of Copenhagen\9. Semester\Seminar financial economtri"
os.makedirs(folder_path, exist_ok=True)

# 2. Parameters
start_date = "2010-01-01"
end_date = datetime.today().strftime('%Y-%m-%d')

# Option 1: The Index (Perfect Price History, No Volume)
ticker_index = "^OMXC20"

# Option 2: The ETF (Real Volume, but starts ~2012)
# Correct Ticker: SPIC25KL.CO (Sparindex OMX C25 KL)
ticker_etf = "SPIC25KL.CO" 

print("--- Downloading Comparison Data ---")

try:
    # Download Index Data
    print(f"1. Fetching INDEX {ticker_index} (2010-Today)...")
    df_index = yf.download(ticker_index, start=start_date, end=end_date, progress=False)
    
    # Download ETF Data
    print(f"2. Fetching ETF {ticker_etf} (Check Start Date)...")
    df_etf = yf.download(ticker_etf, start=start_date, end=end_date, progress=False)

    # Clean up MultiIndex headers if present
    for df in [df_index, df_etf]:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index)

    # Save Index File
    path_index = os.path.join(folder_path, "danish_index_prices_full_history.xlsx")
    df_index.to_excel(path_index)
    
    # Save ETF File
    path_etf = os.path.join(folder_path, "danish_etf_volume_partial_history.xlsx")
    df_etf.to_excel(path_etf)

    print("-" * 30)
    print("REPORT:")
    print(f"Index ({ticker_index}):")
    print(f"   - Start Date: {df_index.index[0].date()}")
    print(f"   - Rows: {len(df_index)}")
    print(f"   - Volume Available? {'NO (Mostly 0)' if df_index['Volume'].sum() == 0 else 'YES'}")
    print(f"   - File: {path_index}")
    
    print(f"\nETF ({ticker_etf}):")
    print(f"   - Start Date: {df_etf.index[0].date()} (Likely 2012)")
    print(f"   - Rows: {len(df_etf)}")
    print(f"   - Volume Available? {'YES' if df_etf['Volume'].sum() > 0 else 'NO'}")
    print(f"   - File: {path_etf}")
    print("-" * 30)

except Exception as e:
    print(f"An error occurred: {e}")
