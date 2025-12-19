import yfinance as yf
import pandas as pd
import os
from datetime import datetime

# ==========================================
# --- 1. CONFIGURATION & PATHS ---
# ==========================================
# Local directory for data persistence. 
# Note: Hardcoded paths should be updated if deployed to a new environment.
FOLDER_PATH = r"C:\Users..."
FILE_NAME = "danish_dataset_ROBUST_BONDS.xlsx"
FULL_PATH = os.path.join(FOLDER_PATH, FILE_NAME)

START_DATE = "2010-01-01"
END_DATE = datetime.today().strftime('%Y-%m-%d')

print(f"--- Initiating Data Aggregation Pipeline ---")
print(f"Target Output: {FULL_PATH}")

# ==========================================
# --- 2. ASSET SPECIFICATIONS ---
# ==========================================

# A. Target Variable: Danish Equity Index
# ^OMXCPI is the OMX Copenhagen Price Index (Benchmark for Danish Market)
DK_TICKER = "^OMXCPI"

# B. Control Variables (Global Macro Factors)
# Standard controls for small open economies: US Market, Exchange Rate, Oil, Volatility.
CONTROLS = {
    "US_SP500":  "^GSPC",   # Proxy for global equity risk
    "FX_USDDKK": "DKK=X",   # Exchange rate risk (USD denominated)
    "Oil_Brent": "BZ=F",    # Energy prices/Inflation proxy
    "VIX":       "^VIX"     # Global risk aversion proxy
}

# C. Bond Yield Proxies (Robustness Strategy)
# Direct historical yield data is often inaccessible via free APIs.
# We utilize German Government Bond ETFs as liquid proxies for European risk-free rates.
# Inverse Relationship Note: Price UP implies Yield DOWN.
BOND_PRIORITIES = {
    "Bond_DE_Long_10Y": ["IBGL.DE", "DB10.DE", "EXVA.DE"], # Proxies: iShares/Xtrackers Bunds
    "Bond_DE_Short_2Y": ["IBGS.DE", "DB2.DE", "EUN3.DE"]   # Proxies: Schatz ETFs
}

# ==========================================
# --- 3. EXTRACTION LOGIC ---
# ==========================================
data_frames = []

def fetch_robust(name, candidates):
    """
    Iteratively attempts to fetch data from a list of candidate tickers.
    Crucial for Bond ETFs where liquidity or listing changes may invalidate specific tickers.
    """
    for ticker in candidates:
        print(f"   Attempting fetch for [{name}] via ticker: {ticker}...")
        try:
            df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
            
            if not df.empty:
                # Handling API Version Differences:
                # yfinance v0.2+ returns MultiIndex columns. We flatten this to ensure consistency.
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                # Standardize Column Names
                col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
                clean_df = df[[col]].rename(columns={col: name})
                
                print(f"      -> Success: Retrieved data for {ticker}")
                return clean_df
        except Exception:
            continue # Silently fail and try the next candidate
            
    print(f"      -> WARNING: All candidates failed for {name}")
    return None

# --- PHASE 1: Target Variable Extraction ---
print("1. Fetching Target Index (Denmark)...")
try:
    df_dk = yf.download(DK_TICKER, start=START_DATE, end=END_DATE, progress=False)
    if not df_dk.empty:
        # Flatten MultiIndex if present
        if isinstance(df_dk.columns, pd.MultiIndex):
            df_dk.columns = df_dk.columns.get_level_values(0)
            
        # Prefix columns to identify them easily in the final dataset
        df_dk = df_dk.add_prefix("DK_")
        data_frames.append(df_dk)
        print(f"   -> Successfully retrieved Target Index.")
    else:
        print("   -> CRITICAL WARNING: Target Index is empty.")
except Exception as e:
    print(f"   -> ERROR fetching Target: {e}")

# --- PHASE 2: Control Variables ---
print("2. Fetching Macroeconomic Controls...")
for name, ticker in CONTROLS.items():
    res = fetch_robust(name, [ticker])
    if res is not None:
        data_frames.append(res)

# --- PHASE 3: Bond Proxies ---
print("3. Fetching Bond Proxies (with Fallback Logic)...")
for name, candidates in BOND_PRIORITIES.items():
    res = fetch_robust(name, candidates)
    if res is not None:
        data_frames.append(res)

# ==========================================
# --- 4. DATA ALIGNMENT & EXPORT ---
# ==========================================
if data_frames:
    print("4. Merging and Aligning Time Series...")
    
    # Outer Join on Date Index
    master_df = pd.concat(data_frames, axis=1)
    
    # Temporal Alignment:
    # We reindex to 'Business Days' ('B') to handle differing holiday schedules 
    # between US, German, and Danish markets (non-synchronous trading).
    master_df.index = pd.to_datetime(master_df.index)
    master_df = master_df.reindex(pd.date_range(start=START_DATE, end=END_DATE, freq='B'))
    master_df.index.name = "Date"
    
    # Convert index to date objects for clean Excel export (removes H:M:S)
    master_df.index = master_df.index.date

    # Data Quality Trim:
    # We remove historical periods where the Target Variable (DK Index) is missing.
    # Prediction is impossible without the target, rendering earlier control data irrelevant.
    dk_cols = [c for c in master_df.columns if c.startswith("DK_")]
    if dk_cols:
        # Find the first valid index (non-NaN) for the Danish data
        valid_mask = master_df[dk_cols].notna().any(axis=1)
        if valid_mask.any():
            first_valid = valid_mask.idxmax()
            print(f"   -> Trimming sparse history. Valid dataset starts: {first_valid}")
            master_df = master_df.loc[master_df.index >= first_valid]

    # Save to Disk
    os.makedirs(FOLDER_PATH, exist_ok=True)
    master_df.to_excel(FULL_PATH)
    
    print("-" * 30)
    print(f"DATA PIPELINE COMPLETE. Dataset saved to:\n{FULL_PATH}")
    print("-" * 30)
    print("Features Captured:")
    print(master_df.columns.tolist())

else:
    print("CRITICAL ERROR: No data could be retrieved. Pipeline aborted.")