import yfinance as yf
import pandas as pd
import os
from datetime import datetime

# 1. Setup Path
folder_path = r"C:\Users\hunne\OneDrive - University of Copenhagen\9. Semester\Seminar financial economtri"
file_name = "danish_etf_full_OHLCV_with_SPY.xlsx"
full_path = os.path.join(folder_path, file_name)

start_date = "2010-01-01"
end_date = datetime.today().strftime('%Y-%m-%d')

print(f"--- Downloading ETF Dataset (SPY + Full Danish OHLCV) ---")

# 2. Define Tickers

# A. TARGET: Danish ETF (We want FULL Open/High/Low/Close/Volume)
#    Ticker: SPIC25KL.CO (Sparindex OMX C25)
dk_ticker = "SPIC25KL.CO"

# B. CONTROL ETFs & VARIABLES (We only want the 'Adj Close' price)
control_tickers = {
    "US_ETF_SP500":     "SPY",          # SPDR S&P 500 ETF Trust (Replaces ^GSPC)
    "EU_ETF_Stoxx50":   "EUE.DE",       # iShares EURO STOXX 50 ETF
    "FX_USDDKK":        "DKK=X",        # Exchange Rate
    "Oil_Brent":        "BZ=F",         # Brent Crude
    "VIX":              "^VIX"          # VIX Index
}

# C. Bond Candidates (Robust Backups)
bond_priorities = {
    "Bond_DE_Long_10Y": ["IBGL.DE", "DB10.DE", "EXVA.DE"], # iShares / Xtrackers
    "Bond_DE_Short_2Y": ["IBGS.DE", "DB2.DE", "EUN3.DE"]   # iShares / Xtrackers
}

data_frames = []

# --- PART 1: DANISH ETF (FULL DATA) ---
print(f"1. Fetching FULL OHLCV for Danish ETF [{dk_ticker}]...")
try:
    # auto_adjust=True ensures Open/High/Low are ALSO adjusted for dividends/splits
    df_dk = yf.download(dk_ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
    
    if not df_dk.empty:
        # Flatten Header
        if isinstance(df_dk.columns, pd.MultiIndex):
            df_dk.columns = df_dk.columns.get_level_values(0)
        
        # Prefix columns to identify them (e.g. DK_Open, DK_Volume)
        df_dk = df_dk.add_prefix("DK_")
        
        data_frames.append(df_dk)
        print(f"   -> Success. Got {len(df_dk)} rows of full data.")
    else:
        print("   -> WARNING: Danish ETF returned no data.")
except Exception as e:
    print(f"   -> ERROR on Denmark: {e}")


# --- PART 2: US ETF & CONTROLS (CLOSE ONLY) ---
print("2. Fetching US ETF & Controls (Adj Close Only)...")

def fetch_close_only(name, ticker_list):
    if isinstance(ticker_list, str): ticker_list = [ticker_list]
    
    for ticker in ticker_list:
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                # Since auto_adjust=True, 'Close' IS the Adjusted Close
                col = 'Close'
                
                clean_df = df[[col]].rename(columns={col: name})
                print(f"   -> Got {name} [{ticker}]")
                return clean_df
        except:
            continue
    print(f"   -> FAILED to find data for {name}")
    return None

# Fetch Standard Controls
for name, ticker in control_tickers.items():
    res = fetch_close_only(name, ticker)
    if res is not None:
        data_frames.append(res)

# Fetch Bonds
for name, candidates in bond_priorities.items():
    res = fetch_close_only(name, candidates)
    if res is not None:
        data_frames.append(res)


# --- PART 3: MERGE & SAVE ---
if data_frames:
    print("3. Merging Data...")
    master_df = pd.concat(data_frames, axis=1)
    
    # Align to Business Days
    master_df.index = pd.to_datetime(master_df.index)
    master_df = master_df.reindex(pd.date_range(start=start_date, end=end_date, freq='B'))
    master_df.index.name = "Date"
    master_df.index = master_df.index.date

    # Trim to Start of Danish Data
    # We look for any valid data in the Danish Open/Close/High columns
    dk_cols = [c for c in master_df.columns if c.startswith("DK_")]
    if dk_cols:
        first_valid = master_df[dk_cols].notna().any(axis=1).idxmax()
        print(f"   -> Trimming dataset to start of Danish ETF data: {first_valid}")
        master_df = master_df.loc[master_df.index >= first_valid]

    os.makedirs(folder_path, exist_ok=True)
    master_df.to_excel(full_path)

    print("-" * 30)
    print(f"SUCCESS. Saved to:\n{full_path}")
    print("-" * 30)
    print("Columns Included:")
    print(master_df.columns.tolist())
else:
    print("CRITICAL ERROR: No data collected.")