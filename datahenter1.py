import yfinance as yf
import pandas as pd
import os
from datetime import datetime

# 1. Setup Path
folder_path = r"C:\Users\hunne\OneDrive - University of Copenhagen\9. Semester\Seminar financial economtri"
file_name = "danish_dataset_ROBUST_BONDS.xlsx"
full_path = os.path.join(folder_path, file_name)

start_date = "2010-01-01"
end_date = datetime.today().strftime('%Y-%m-%d')

print(f"--- Downloading Dataset with ROBUST Bond Backups ---")

# 2. Define Variable Groups

# A. Danish Index (Full OHLCV)
dk_ticker = "^OMXCPI"

# B. Control Variables (Standard Close Price)
controls = {
    "US_SP500":  "^GSPC",
    "FX_USDDKK": "DKK=X",
    "Oil_Brent": "BZ=F",
    "VIX":       "^VIX"
}

# C. Bond Candidates (Priority List)
# The script will try these one by one until it finds data.
# Logic: Prices of German Govt Bond ETFs (Price UP = Yield DOWN)
bond_priorities = {
    "Bond_DE_Long_10Y": ["IBGL.DE", "DB10.DE", "EXVA.DE"], # iShares, Xtrackers, Deka
    "Bond_DE_Short_2Y": ["IBGS.DE", "DB2.DE", "EUN3.DE"]   # iShares, Xtrackers, iShares Euro
}

data_frames = []

# --- FUNCTION: Robust Downloader ---
def fetch_robust(name, candidates):
    for ticker in candidates:
        print(f"   Attempting {name} via [{ticker}]...")
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if not df.empty:
                # Clean and Return
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
                clean_df = df[[col]].rename(columns={col: name})
                print(f"      -> SUCCESS with {ticker}!")
                return clean_df
        except:
            pass # Try next
    print(f"      -> FAILED: All candidates for {name} were empty.")
    return None

# --- PART 1: DK INDEX (Full OHLCV) ---
print("1. Fetching Denmark Index (Full Data)...")
try:
    df_dk = yf.download(dk_ticker, start=start_date, end=end_date, progress=False)
    if not df_dk.empty:
        if isinstance(df_dk.columns, pd.MultiIndex):
            df_dk.columns = df_dk.columns.get_level_values(0)
        df_dk = df_dk.add_prefix("DK_")
        data_frames.append(df_dk)
        print(f"   -> Got Denmark Index.")
    else:
        print("   -> WARNING: Denmark Index empty.")
except Exception as e:
    print(f"   -> ERROR: {e}")

# --- PART 2: CONTROLS ---
print("2. Fetching Controls...")
for name, ticker in controls.items():
    res = fetch_robust(name, [ticker])
    if res is not None:
        data_frames.append(res)

# --- PART 3: BONDS (With Backups) ---
print("3. Fetching Bonds (Using Backup Lists)...")
for name, candidates in bond_priorities.items():
    res = fetch_robust(name, candidates)
    if res is not None:
        data_frames.append(res)

# --- PART 4: MERGE ---
if data_frames:
    print("4. Merging and Saving...")
    master_df = pd.concat(data_frames, axis=1)
    
    # Align dates
    master_df.index = pd.to_datetime(master_df.index)
    master_df = master_df.reindex(pd.date_range(start=start_date, end=end_date, freq='B'))
    master_df.index.name = "Date"
    master_df.index = master_df.index.date

    # Trim to valid DK data start
    dk_cols = [c for c in master_df.columns if c.startswith("DK_")]
    if dk_cols:
        first_valid = master_df[dk_cols].notna().any(axis=1).idxmax()
        print(f"   -> Trimming empty history. Data starts: {first_valid}")
        master_df = master_df.loc[master_df.index >= first_valid]

    os.makedirs(folder_path, exist_ok=True)
    master_df.to_excel(full_path)
    
    print("-" * 30)
    print(f"SUCCESS. File saved to:\n{full_path}")
    print("-" * 30)
    print("Columns captured:")
    print(master_df.columns.tolist())
else:
    print("CRITICAL ERROR: No data found.")