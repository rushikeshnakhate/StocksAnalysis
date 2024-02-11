from pathlib import Path
from datetime import date
from jugaad_data.nse import NSELive
from jugaad_data.nse import bhavcopy_save
import pandas as pd
output_dir = Path("/content/sample_data")
import os




def validate_required_args(kwargs, *required_args):
    for arg in required_args:
        if arg not in kwargs:
            raise ValueError(f"Error: '{arg}' argument not provided, actual received: {kwargs}")

import pickle

def load_or_download_and_cache_data(download_function:callable,**kwargs):
  validate_required_args(kwargs, 'cache_file')
  cache_file : str = kwargs.get("cache_file")

  if os.path.exists(cache_file):
    with open(cache_file,'rb') as file:
      return pickle.load(file)

  data_df = download_function(**kwargs)
  # if not data_df.empty:
  with open(cache_file,'wb') as file:
    pickle.dump(data_df,file)
  return data_df

import os
def download_bhavcopy(**kwargs):
  """
  # bhavcopy_save: Generates a CSV file with a random name.
  # The expected name for the downloaded Bhavcopy file is "bhavcopy_${start_date}.csv".
  # The cache file is named as "bhavcopy_${start_date}.pkl". A renaming process is required.
  """
  try:
    validate_required_args(kwargs, 'cache_file','start_date')
    cache_file: str = kwargs.get("cache_file")
    start_date : date= kwargs.get("start_date")
    generated_csv_filename = bhavcopy_save(start_date, output_dir)
    cache_file_to_csv_name = cache_file.with_suffix(".csv")
    os.rename(generated_csv_filename,cache_file_to_csv_name)
    return pd.read_csv(cache_file_to_csv_name)
  except Exception as ex:
    print("downlod bhavcopy error={}".format(ex))




# Get stocks
def fetch_or_genenate_stocks(start_date: date):
  output_file  = Path(output_dir)/("NseData_" + str(start_date) +".pkl")
  return load_or_download_and_cache_data(download_bhavcopy,start_date= start_date,cache_file=output_file)


# Get Sectors
import yfinance as yf
import concurrent.futures
import threading

def populate_sector(tick: str):
  tick_ = tick + ".NS"
  try:
    ticker = yf.Ticker(tick_)
    # print("populate_sector sector for sym={}".format(tick))
    return ticker.info['sector']
  except Exception as e:
    return "NOT_FOUND_ON_YAHOO"


def download_sectors(**kwargs):
  validate_required_args(kwargs, 'stocks')
  stocks_df : pd.Dataframe = kwargs.get('stocks')
  new_data = []
  for symbol in stocks_df['SYMBOL']:
      sector = populate_sector(symbol)
      new_data.append((symbol, sector))
  return pd.DataFrame(new_data, columns=['SYMBOL', 'SECTOR'])


# lock = threading.Lock()  # Create a lock for thread safety
# results = {}  # Use a dictionary to store results with symbol as key
# def process_stock(symbol):
#   # Acquire the lock to prevent race conditions
#   with lock:
#     sector = populate_sector(symbol)
#     results[symbol] = sector

# def download_sectors_concurrently(**kwargs):
#   validate_required_args(kwargs, 'stocks')
#   stocks_df : pd.Dataframe = kwargs.get('stocks')
#   with concurrent.futures.ThreadPoolExecutor() as executor:
#       executor.map(process_stock, stocks_df['SYMBOL'].tolist())
#   dd1 = pd.DataFrame(results)
#   print("00000000000000000")
#   display(dd1)
#   print("00000000000000000")
#   return dd1


# def fetch_or_generate_sector_to_stocks(stocks : pd.DataFrame):
#   output_file  = Path(output_dir)/("Sectors.pkl")
#   stocks_to_sector_map_df = load_or_download_and_cache_data(download_sectors,cache_file=output_file,stocks=stocks)
#   dd =  pd.merge(stocks, stocks_to_sector_map_df, on='SYMBOL', how='inner',suffixes=('', ''))
#   if dd.empty:
#     raise ValueError("Error: The DataFrame is empty.")
#   return dd


def fetch_or_generate_sector_to_stocks(stocks : pd.DataFrame):
  output_file  = Path(output_dir)/("Sectors.pkl")
  stocks_to_sector_map_df = load_or_download_and_cache_data(download_sectors,cache_file=output_file,stocks=stocks)
  try:
   return pd.merge(stocks, stocks_to_sector_map_df, on='SYMBOL', how='inner',suffixes=('', ''))
  except pd.errors.MergeError as e:
    print(f"Merge failed: {e}")


# Calculate Performace
def calculate_performace(stocks_df : pd.DataFrame, start_date: date):
    # Calculate daily returns
  print("calculate_performace for date={}".format(start_date))
  stocks_df['DAILY_RETURN'] =(stocks_df['CLOSE'] - stocks_df['OPEN'])/ stocks_df['OPEN']

  # Group by 'SECTOR' and calculate average daily return and volatility
  sector_performance_df = stocks_df.groupby('SECTOR').agg({
      # stocks_list=('STOCK', lambda x: list(x))
      'DAILY_RETURN': ['std','mean','count']
  }).reset_index()

  sector_performance_df["TRADEDATE"] = start_date

  # Flatten the column names, but only for columns where the second part is not an empty string
  sector_performance_df.columns = [
      f"{col[0]}_{col[1]}" if col[1] else col[0] for col in sector_performance_df.columns
  ]
  return sector_performance_df


def prepare_stocks_data_and_calculate_performance(**kwargs):
  try:
    validate_required_args(kwargs, 'start_date')
    start_date : date = kwargs.get('start_date')
    stocks_df = fetch_or_genenate_stocks(start_date)
    stocks_df_with_sector = fetch_or_generate_sector_to_stocks(stocks_df)
    return calculate_performace(stocks_df_with_sector,start_date)
  except ValueError as e:
    print(e)

def fetch_or_generate_performance_for_stocks(start_date:date):
  sector_performance_output_file = Path(output_dir)/("SectorPeformance_" + str(start_date) +".pkl")
  return load_or_download_and_cache_data(prepare_stocks_data_and_calculate_performance,start_date=start_date,cache_file=sector_performance_output_file)


# main Processs
#from google.colab import data_table
#from datetime import date, timedelta#
# import jugdad

#data_table.enable_dataframe_formatter()

if __name__ == "__main__":
  # from nsetools import Nse
  # holidays = load_calendar_holidays("nseindia")

  # # Get NSE holidays for the specified year
  # year = 2024
  # nse_holidays = holidays(year)
  # display(nsetools)

  start_date = date (2024,1,15)
  df = fetch_or_generate_performance_for_stocks(start_date)
  print(df)

  start_date = date (2024,2,9)
  df = fetch_or_generate_performance_for_stocks(start_date)
  print(df)
