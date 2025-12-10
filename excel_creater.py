import yfinance as yf
import pandas as pd
from datetime import date

ticker = "^OMXC20"
start_date = "2010-01-01"  # Or start="2015-10-17" if you only wanted 10 years
end_date = date.today().strftime("%Y-%m-%d")

print(f"Downloading daily data for {ticker} from {start_date} to {end_date}...")
omx_data_daily = yf.download(ticker, start=start_date, end=end_date, interval="1d")

# --- NEW INSPECTION CODE ---
if omx_data_daily.empty:
    print(f"❌ Initial download is EMPTY. Yahoo Finance likely doesn't have data for {ticker} over this period.")
else:
    print(f"✅ Initial download succeeded! Received {len(omx_data_daily)} daily data points.")
    print("\n--- Preview of RAW Daily Data ---")
    print(omx_data_daily.head()) # Check the start date
    print(omx_data_daily.tail()) # Check the end date

    # --- Proceed with Resampling ---
    close_prices = omx_data_daily[['Close']]
    # Use 'ME' (Month End)
    month_end_prices = close_prices.resample('ME').last()
    month_end_prices_clean = month_end_prices.dropna()
    
    # ... rest of your saving code ...
    filename = "omxc20_monthly_prices.xlsx"
    df_to_save = month_end_prices_clean.rename(columns={'Close': 'Adjusted Month-End Close'})
    df_to_save.to_excel(filename)
    
    print(f"\n✅ Success! Data has been saved to '{filename}'. Total **{len(df_to_save)}** month-end prices.")
    print(df_to_save.head(5))
    print(df_to_save.tail(5))
