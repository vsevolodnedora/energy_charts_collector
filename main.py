"""
    Fetch data from https://www.energy-charts.info/
    Inspired by this [repo](https://github.com/chris1869/grid_data)

    Germany is chosen as Energy charts have the largest amount of information for it.
    The time period is chosen to be 2015-2024 as earlier data is not always available.

    The code scrapes the data from 'highcharts' of the webpage as most of the data required
    is not available via API calls or too cumbersome to get via direct download.
    We use selenium for headless scraping.

    The code fetches data for the following pages and produces the following data files:
    - __Power__ (power)
        - `DE_power_entsoe_2015_2024` data from ENTSO-E
        - `DE_power_public_2015_2024` data from Public Energy
        - `DE_power_sw_2015_2024` data for solar and wind generation from different TSOs
        - `DE_power_tcs_saldo_2015_2024` data for cross-border trade between Germany and its neighbours
        - `DE_power_total_2015_2024` data for total load / production (aka ENTSO-E)
    - __Prices__ (price_spot_market)
        - `DE_price_spot_market_2015_2024` spot electricity market prices
        - `DE_price_volume_dayahead_2015_2024` day-ahead auction prices and volumes
        - `DE_price_volume_intraday_2015_2024` intraday continous trade: prices and volumes
    - __Weather__ (climate_hours)
        `DE_weather_air_humidity_2015_2024` air humidity
        `DE_weather_air_temperature_2015_2024` air temperature
        `DE_weather_air_solar_diffuse_2015_2024` solar diffuse
        `DE_weather_air_solar_globe_2015_2024` solar globe
        `DE_weather_air_wind_direction_2015_2024` wind direction
        `DE_weather_air_wind_speed_2015_2024` wind speed

    The weather data is provided for all available weather stations (see weather_stations.csv for
    their coordinates and elevation)

    NOTE: due to licencing data itself cannot be provided here.
    Thus at least one run of the script is required to fetch it.

    NOTE: because yearly 'highcharts' can be very busy, loading takes time and
    sometimes fails. Several methods to overcome it are implemented. However, sometimes,
    re-running the script helps (it will skip over already downloaded and cached data)

    NOTE: over the years trading zones have changed. DE-AT-LU became DE-LU and then DE in some
    markets. We simplify it all here by removing them everywhere using mapping.

    NOTE: Power data is given with 15min frequency and thus might require downsamling to be
    aligned with the rest. Also, no usual aggregation is done for power generation.

    NOTE: climate data is scraped differently from the others as it is not available for the
    whole year for hourly intervals. Thus, the code scrape it for each month of a given year
    and then concatenate the result. Also note, that the climate data is available differently
    for different cities in Germany and only the data that has number of entries expected for
    the hourly data is stored.
"""

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import StaleElementReferenceException
from statsmodels.sandbox.distributions.genpareto import shape
from webdriver_manager.chrome import ChromeDriverManager

import pandas as pd
import hashlib
import os
import gc
import json
import time
import calendar
import pycountry
from collections import Counter
from datetime import datetime, timedelta

''' -------------- energy-charts URLs -------------------- '''

def create_energy_url(year=2024, country:str='DE', source="public"):
    """
    Creates an URL pointing to the relevant subsection of energy-charts.info based on the provided year and source.

    Args:
        year (int): The year for which the energy data is requested. Defaults to 2024.
        source (str): The data source for the energy data. Defaults to "public".
        example:
            https://www.energy-charts.info/charts/power/chart.htm?l=en&c=DE&year=2024&source=total&legendItems=2wfw4
    Returns:
        str: The constructed energy URL.
    """
    base_url = "https://www.energy-charts.info/charts/power/chart.htm?l=en"
    return base_url + f"&c={country}&source={source}&interval=year&year={year}&timezone=utc"

def create_market_url(year=2024,market:str="price_spot_market",country:str='DE', period:None or str=None):
    """
    Creates an URL pointing to the relevant subsection of energy-charts.info based on the provided year market,
    and period

    Args:
        year (int): The year for which the energy data is requested. Defaults to 2024.
        source (str): The data source for the energy data. Defaults to "public".

    Returns:
        str: The constructed energy URL.
    """
    base_url = f"https://www.energy-charts.info/charts/{market}/chart.htm?l=en"
    url = base_url + f"&c={country}&interval=year&year={year}&timezone=utc"
    if period is not None:
        url += "&period=" + period
    return url

def create_weather_url(year=2024,month:str='01',country:str='DE', source:str='wind_speed'):
    """
    Creates an URL pointing to the relevant subsection of energy-charts.info based on the provided year
    month and source.

    Args:
        year (int): The year for which the energy data is requested. Defaults to 2024.
        source (str): The data source for the energy data. Defaults to "public".

    Returns:
        str: The constructed energy URL.
    """
    base_url = f"https://www.energy-charts.info/charts/{'climate_hours'}/chart.htm?l=en"
    url = base_url + f"&c={country}&interval=month&year={year}&source={source}&month={month}&timezone=utc"
    return url

''' -------------- selenium helper functions --------------------- '''

def safe_webdriver_initialization()-> webdriver.Chrome:
    """ create headless selenium driver """
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--window-size=1920,1080')  # Add a default window size
    options.add_argument('--disable-gpu')  # Disable GPU hardware acceleration
    options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3')

    try:
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()),options=options)
    except Exception as e:
        print(f"Error initializing WebDriver: {e}")
        driver = None
    return driver

def close_driver(driver:webdriver.Chrome):
    if driver:
        driver.delete_all_cookies()  # Clear all cookies
        driver.quit()

def extract_data_headless(url:str, wait_for_html_load:bool,sleep_time:int, script_option:str, cache_fname:str) -> None:
    """

    Open webpage in the headless mode (no GUI) and extract data from 'highcharts' by executing
    the java script on it. Note, there are many ways how this function may fail due to incliplete
    page load, so several try-catch blocks are implemented.

    However, if the code fails here, it might still help to rerun it, as

    :param url:
    :param wait_for_html_load:
    :param sleep_time:
    :param script_option:
    :param cache_fname:
    :return:
    """
    driver = safe_webdriver_initialization()

    # with safe_webdriver_initialization() as driver:

    if driver is None:
        raise BaseException("WebDriver not initialized")

    driver.get(url)

    if wait_for_html_load:
        print(f"Waiting for HTML load for {sleep_time}...")
        # wait = WebDriverWait(driver, 20)  # Wait for up to 20 seconds
        wait = WebDriverWait(driver, sleep_time)
        _ = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.highcharts-container")))
        # del _

    # sleep to allow extra time for loading
    time.sleep(sleep_time)

    # select script to run on the page
    if script_option == 'simple':
        script = """
                series_vals = {};
                Highcharts.charts[0].series.forEach(function(s){series_vals[s.name] = [];s.data.forEach(function(d) {series_vals[s.name].push(d.y)});});
                return series_vals;
            """
    elif script_option == 'advanced':
        script = """
                if (!Highcharts.charts[0] || !Highcharts.charts[0].series) {
                    return {};
                }
                var series_vals = {};
                Highcharts.charts[0].series.forEach(function(s) {
                    if (s && s.data) {
                        series_vals[s.name] = [];
                        s.data.forEach(function(d) {
                            series_vals[s.name].push(d.y);
                        });
                    }
                });
                return series_vals;
            """
    else:
        raise NameError(f"Script option {script_option} not recognized")

    # run script to extract data from the highcharts
    print(f"Executing java script option={script_option}")
    json_data = driver.execute_script(script)

    # write the result to cach file
    print(f"Saving cache data {cache_fname}")
    with open(cache_fname, "w") as f:
        json.dump(json_data, f)

    close_driver(driver=driver)

    return None

def check_json_file_for_monthly_data(cache_fname:str) -> pd.DataFrame or None:
    # Load JSON data from a file
    with open(cache_fname, 'r') as file:
        data = json.load(file)

    # Dictionary to hold the lengths of each list
    lengths = {}

    # Collect the lengths of each list
    for key, value in data.items():
        if isinstance(value, list):
            lengths[key] = len(value)
        else:
            lengths[key] = None  # Use None for non-list data

    # Find the most common length
    length_counter = Counter(lengths.values())
    most_common_length = length_counter.most_common(1)[0][0]  # This returns the most common length

    # Print keys corresponding to the most common length
    keys_with_most_common_length = []
    for key, length in lengths.items():
        if length == most_common_length:
            keys_with_most_common_length.append(key)
    print(f"Keys with the most common length ({most_common_length}) "
          f"N={len(keys_with_most_common_length)}/{len(lengths)}: "
          f"{keys_with_most_common_length}")

    # Print keys with lengths different from the most common length
    keys_with_different_length = []
    for key, length in lengths.items():
        if length != most_common_length:
            keys_with_different_length.append(key)
    print(f"Keys with different lengths: "
          f"N={len(keys_with_different_length)}/{len(lengths)}: {keys_with_different_length}")

    # Function to calculate the number of hours in a month, adjusting for current date if necessary
    def hours_in_month(year:int, month:int):
        now = datetime.now()
        if ((year == now.year) and (month == now.month)):
            # Calculate hours up to the current hour
            return (now.day - 1) * 24 + now.hour
        else:
            num_days = calendar.monthrange(year, month)[1]
            return 24 * num_days

    # Calculate the number of expected hourly entries
    pices = cache_fname.split('_')
    month,year=int(pices[3]),int(pices[2])
    expected_hours = hours_in_month(year,month)
    # Compare most common length with expected number of hourly entries
    print(f"Expected number of hourly entries for {calendar.month_name[month]} {year}: {expected_hours}")
    print(f"Does the most common length match the expected hours? {'Yes' if most_common_length == expected_hours else 'No'}")


    # Function to generate hourly timestamps for a given month and year, adjusted for current time
    def generate_timestamps(year:int, month:int, num_hours:int or None = None):
        now = datetime.now()
        num_days = calendar.monthrange(year, month)[1]
        start_date = datetime(year, month, 1)
        if year == now.year and month == now.month:
            end_date = start_date + timedelta(hours=num_hours-1)
            # end_date = datetime(year, month, now.day, now.hour)  # up to the current hour
        else:
            end_date = datetime(year, month, num_days, 23)
        return pd.date_range(start=start_date, end=end_date, freq='h')

    # Filter keys with the most common length
    valid_keys = [key for key, length in lengths.items() if length == most_common_length]
    # Creating the DataFrame
    df_data = {key: data[key] for key in valid_keys}
    df = pd.DataFrame(df_data)

    # Compare most common length with expected number of hourly entries
    if most_common_length == expected_hours:
        # Set the timestamp index
        df.index = generate_timestamps(year, month, None)

    else:
        print(f"Warning! Most common length does not match expected number of hourly entries. "
              f"Creating dataframe for {most_common_length} hourly timestamps")
        # Set the timestamp index
        df.index = generate_timestamps(year, month, most_common_length)
    return df

def fetch_or_load_json_for_url_request(url:str, cache_fname:str) -> str:
    """
    Uses selenium to load the webpage in the headless mode (no GUI) and
    tries to extract the data from it calling 'extract_data_headless'

    Due to many, many issues I encoutered when trying to read the data from these
    `highcharts` several try-catch blocks are implmeneted to allow the code to
    wait longer for the data to be loaded on the webpage

    :param url:
    :param cache_fname:
    :return:
    """

    cache_fname = f"cache/{cache_fname}.json"
    print(f"Caching data {cache_fname}")
    if os.path.exists(cache_fname):
        print("Cache file already exists. Reading... ", url)

        return cache_fname
    else:
        print("Cache file deos not exists. Fetching... ", url)

    max_attempts = 5
    attempt = 0
    sleep_time = 5

    while attempt < max_attempts:
        try:
            extract_data_headless(url,True, sleep_time,script_option='simple',cache_fname=cache_fname)
            break
        except Exception as e:
            print(f"Attempt {attempt + 1}/{max_attempts} (sleep_time={sleep_time}) failed with {str(e)}")
            attempt += 1
            sleep_time += 10  # Increase sleep time for each attempt
        gc.collect()

    if not os.path.exists(cache_fname):
        raise FileNotFoundError("Cache file not found. Failed to fetch the data. See errors above.")

    print("Successfully extracted data after {} attempts.".format(attempt))

    # df = pd.read_json(cache_fname)
    return cache_fname

def get_index_serie_for_the_year(year, df):
    """
    Creates a list of timestamps for a year
    For current year it generates timestamps up to today.
    The function also checks if the number of timestamps agrees with the
    number of datapoins


    :param year:
    :param df:
    :return:
    """
    # year = datetime.today().year  # Assuming the data is for this year

    # Calculate days based on whether the data is for this year or a full past year
    today = datetime.today()
    if year == today.year:
        start_of_year = datetime(today.year, 1, 1)
        elapsed_days = (today - start_of_year).days + 1  # Including today
    else:
        elapsed_days = 365  # For a non-leap year or adjust for leap years if required

    # Calculate expected rows for each frequency
    hourly_rows_expected = 24 * elapsed_days
    quarter_hourly_rows_expected = 24 * 4 * elapsed_days

    # Detect frequency of dataframe
    shape0 = df.shape[0]
    if abs(shape0 - hourly_rows_expected) <= 24: freq = "1h"
    elif abs(shape0 - quarter_hourly_rows_expected) <= 96: freq = "15min"
    elif (hourly_rows_expected - shape0) == 25:
        freq = "1h"
        print(f"Warning! It seems that the data is missing last 25 hours. Given shape is {shape0}. "
              f"Expected {hourly_rows_expected} for {elapsed_days} elapsed days")
    else:
        raise ValueError(f"Unknown frequency with {shape0} entries. Expected {hourly_rows_expected} "
                         f"for hourly data and {quarter_hourly_rows_expected} "
                         f"for quater-hourly for {elapsed_days} elapsed days")

    # Debugging output
    print(f"Expected rows for hourly data: {hourly_rows_expected}")
    print(f"Expected rows for 15-min data: {quarter_hourly_rows_expected}")
    print(f"Actual data rows: {shape0}")
    print(f"Frequency detected: {freq}")

    index_serie = pd.date_range(start=f"01-01-{year}", periods=shape0, freq=freq)
    return index_serie

def generate_hourly_timestamps(year, month):
    # Starting date at the beginning of the specified month and year
    start_date = f"{year}-{month:02d}-01 00:00:00"
    # Compute the end date by adding one month and then subtracting one hour
    end_date = pd.to_datetime(start_date) + pd.DateOffset(months=1) - pd.DateOffset(hours=1)
    # Generate date range with hourly frequency
    timestamps = pd.date_range(start=start_date, end=end_date, freq='H', tz='UTC')
    # Return as a pandas Series object
    return pd.Series(timestamps)

def prepare_power_dataframe(df, year):

    print(f"Processing dataframe for year {year} df.shape[0]={df.shape[0]}")

    # Set the date range and index
    index_serie = get_index_serie_for_the_year(year, df)

    try:
        df.set_index(index_serie, inplace=True)
    except Exception as e:
        print(f"Error setting index: {e}")

    df["year"] = df.index.year
    df["hour"] = df.index.hour

    return df


def process_dataframe(df,mapping:dict):

    df.rename(columns=mapping, inplace=True)

    # Check for duplicate column names
    if df.columns.duplicated().any():
        # Create a new DataFrame for the result
        new_df = pd.DataFrame(index=df.index)

        # Identify all unique column names
        unique_columns = df.columns.unique()
        for name in unique_columns:
            columns = df.loc[:, name]  # This selects all columns with the same name
            if isinstance(columns, pd.DataFrame):  # Check if multiple columns with the same name exist
                # Find the column with the least NaNs by converting to list of series and comparing NaN counts
                min_nan_column = min(columns.transpose().to_dict().values(), key=lambda col: pd.Series(col).isna().sum())
                new_df[name] = pd.Series(min_nan_column)
            else:
                # If it's a single column, just assign it directly
                new_df[name] = columns

        df = new_df

    # fill nans with interpolation for prices
    for col in df.columns:
        if (df[col].isna().sum() > 0.2 * len(df[col])):
            print(f"Warning for col {col} nans > 20\%")
    df.interpolate(method='linear', limit_direction='both', inplace=True)

    return df


''' ------------------------- '''

def prepare_and_store_power(country, years, area, mapping):
    df = pd.DataFrame()

    # scrape each year
    for i, a_year in enumerate(years):
        print(f"Processing power for year={a_year} area={area}")

        cache_fpath = fetch_or_load_json_for_url_request(
            create_energy_url(year=a_year, country=country, source=area),
            cache_fname=f'{country}_{"power"}_{a_year}_yearly' if area is None
            else f'{country}_{"power"}_{a_year}_{area}_yearly'
        )
        df_i = pd.read_json(cache_fpath)

        df_i = prepare_power_dataframe(df_i, a_year)
        df_i = process_dataframe(df_i, mapping)
        print(df_i.columns)
        df = pd.concat([df, df_i])

    if area is None: res_name = f"output/{country}_{'power'}_{years[0]}_{years[-1]}.csv"
    else: res_name = f"output/{country}_{'power'}_{area}_{years[0]}_{years[-1]}.csv"
    df.to_csv(res_name)

def prepare_and_store_market(country, years, market, period:None or str, mapping:dict):
    df = pd.DataFrame()

    # scrape each year
    for i, a_year in enumerate(years):
        print(f"Processing market for year={a_year} market={market} period={period}")

        cache_fpath = fetch_or_load_json_for_url_request(
            create_market_url(year=a_year, market=market, country=country, period=period),
            cache_fname=f'{country}_{market}_{a_year}_yearly' if period is None
            else f'{country}_{market}_{a_year}_{period}_yearly'
        )
        df_i = pd.read_json(cache_fpath)
        df_i = prepare_power_dataframe(df_i, a_year)
        df_i = process_dataframe(df_i, mapping)
        print(df_i.columns)
        df = pd.concat([df, df_i])

    if period is None: res_name = f"output/{country}_{market}_{years[0]}_{years[-1]}.csv"
    else: res_name = f"output/{country}_{market}_{period}_{years[0]}_{years[-1]}.csv"
    df.to_csv(res_name)

def prepare_and_store_weather(country, years, quantity, mapping:dict):
    df = pd.DataFrame()
    for i, a_year in enumerate(years):
        for j in range(1, 13):
            # avoid going into the future
            now = datetime.now()
            if ((a_year >= now.year) and (j > now.month)):
                break

            month=f"{j:02}"
            try:
                print(f"Processing weather for year={a_year} month={month} quantity={quantity}")
                cache_fpath = fetch_or_load_json_for_url_request(
                    create_weather_url(year=a_year, month=month, country=country, source=quantity),
                    cache_fname=f'{country}_weather_{a_year}_{month}_monthly' if quantity is None
                    else f'{country}_weather_{a_year}_{month}_{quantity}_monthly'
                )
            except FileNotFoundError:
                print(f"Failed to get data for year={a_year} month={month} quantity={quantity}")
                cache_fpath = None

            if cache_fpath is not None:
                df_i = check_json_file_for_monthly_data(cache_fpath)
                df_i=process_dataframe(df_i,mapping=mapping)
                df = pd.concat([df, df_i])
    if quantity is None: res_name = f"output/{country}_weather_{years[0]}_{years[-1]}.csv"
    else: res_name = f"output/{country}_weather_{quantity}_{years[0]}_{years[-1]}.csv"
    df.to_csv(res_name)


def main():
    country = 'DE'
    years = range(2015, 2025)


    # Energy production and load
    mapping = {
        # ----------------------- POWER -----------------
        # original (as of 2015)
        'Biomass' : 'Biomass',
        'Cross border electricity trading' : 'Cross_Border_Electricity_Trading',
        'Day Ahead Auction (DE-AT-LU)':'DA_Auction',
        'Fossil brown coal / lignite':'Fossil_Brown_Coal_Lignite',
        'Fossil coal-derived gas':'Fossil_Brown_Coal_Gas',
        'Fossil gas':'Fossil_Gas',
        'Fossil hard coal':'Fossil_Hard_Coal',
        'Fossil oil':'Fossil_Oil',
        'Geothermal':'Geothermal',
        'Hydro Run-of-River':'Hydro_Run_of_River',
        'Hydro pumped storage':'Hydro_Pumped_Storage',
        'Hydro pumped storage consumption':'Hydro_Pumped_Storage_Consumption',
        'Hydro water reservoir':'Hydro_Water_Reservoir',
        'Load':'Load',
        'Nuclear':'Nuclear',
        'Others':'Others',
        'Renewable share of generation':'Renewable_Share_of_Generation',
        'Renewable share of load':'Renewable_Share_of_Load',
        'Residual load':'Residual_Load',
        'Solar':'Solar',
        'Waste':'Waste',
        'Wind offshore':'Wind_Offshore',
        'Wind onshore':'Wind_Onshore',
        # ---
        'Day Ahead Auction (DE-AT-LU, DE-LU)': 'DA_Auction',
        'Day Ahead Auction (DE-LU)': 'DA_Auction',

        # ------------------- CROSS BORDER TRADE ----------------------
        'Czech Republic' : 'Czech_Republic',

        # ------------------- TOTAL -----------------
        'Load (incl. self-consumption)' : 'Load_Incl_Self_Consumption',

        # --------------------- SW --------------------------
        'Solar 50Hertz':'Solar_50Hertz',
        'Solar Amprion':'Solar_Amprion',
        'Solar TenneT':'Solar_TenneT',
        'Solar TransnetBW':'Solar_TransnetBW',
        'Wind offshore 50Hertz':'Wind_Offshore_50Hertz',
        'Wind offshore TenneT':'Wind_Offshore_TenneT',
        'Wind onshore 50Hertz':'Wind_Onshore_50Hertz',
        'Wind onshore Amprion':'Wind_Onshore_Amprion',
        'Wind onshore TenneT':'Wind_Onshore_TenneT',
        'Wind onshore TransnetBW':'Wind_Onshore_TransnetBW'
    }
    for area in ["public", "tcs_saldo", "total", "entsoe", "sw"]:
        prepare_and_store_power(country, years, area, mapping=mapping)

    # spot electricity market prices and volumes
    market="price_spot_market"
    mapping={
        # pre 2018
        "CO2 Emission Allowances, Auction DE" : "CO2_Emission_Allowance_DE",
        "CO2 Emission Allowances, Auction EU": "CO2_Emission_Allowance_EU",
        "Day Ahead Auction (DE-AT-LU)":"DA_Auction",
        "Intraday Continuous Average Price (DE-AT-LU)":"ID_Continuous_Average_Price",
        "Intraday Continuous High Price (DE-AT-LU)":"ID_Continuous_High_Price",
        "Intraday Continuous Low Price (DE-AT-LU)":"ID_Continuous_Low_Price",
        "Intraday Continuous ID3-Price (DE-AT-LU)" : "ID_Continuous_ID3_Price",
        "Intraday auction, average of the 15 min auctions (DE-AT-LU)":"ID_Auction",
        #
        "Cross border electricity trading":"Cross_Border_Electricity_Trading",
        "Day Ahead Auction (DE-AT-LU, DE-LU)":"DA_Auction",
        "Hydro pumped storage consumption":"Hydro_Pumped_Storage_Consumption",
        "Intraday Continuous Average Price (DE-AT-LU, DE-LU)":"ID_Continuous_Average_Price",
        "Intraday Continuous High Price (DE-AT-LU, DE-LU)":"ID_Continuous_High_Price",
        "Intraday Continuous ID1-Price (DE-AT-LU, DE-LU)":"ID_Continuous_ID1_Price",
        "Intraday Continuous ID3-Price (DE-AT-LU, DE-LU)":"ID_Continuous_ID3_Price",
        "Intraday Continuous Low Price (DE-AT-LU, DE-LU)":"ID_Continuous_Low_Price",
        "Intraday auction, average of the 15 min auctions (DE-AT-LU, DE-LU)":"ID_Auction",
        #
        "Intraday Continuous Average Price (DE-LU)":"ID_Continuous_Average_Price",
        "Day Ahead Auction (DE-LU)":"DA_Auction",
        "Intraday Continuous High Price (DE-LU)":"ID_Continuous_High_Price",
        "Intraday Continuous ID1-Price (DE-LU)":"ID_Continuous_ID1_Price",
        "Intraday Continuous ID3-Price (DE-LU)":"ID_Continuous_ID3_Price",
        "Intraday Continuous Low Price (DE-LU)":"ID_Continuous_Low_Price",
        "Intraday auction, average of the 15 min auctions (DE-LU)":"ID_Auction",
    }
    prepare_and_store_market(country, years, market, None, mapping=mapping)

    # day-ahead electricity market prices and volumes
    market="price_volume"
    mapping={
        "Day Ahead Auction (DE-AT-LU)":"DA_Auction",
        "Day Ahead Auction (DE-LU)":"DA_Auction",
        "Day Ahead Auction (DE-AT-LU, DE-LU)":"DA_Auction",
        # ---
        "Day Ahead Auction (DE-AT-LU), volume":"DA_Auction_Volume",
        "Day Ahead Auction (DE-LU), volume":"DA_Auction_Volume",
        "Day Ahead Auction (DE-AT-LU, DE-LU), volume":"DA_Auction_Volume"
    }
    prepare_and_store_market(country, years, market, "dayahead",mapping=mapping)

    # intraday electricity market prices and volumes
    mapping={
        "Intraday Continuous Average Price (DE-AT-LU)":"ID_Continuous_Average_Price",
        "Intraday Continuous Average Price (DE-LU)":"ID_Continuous_Average_Price",
        "Intraday Continuous Average Price (DE-AT-LU, DE-LU)":"ID_Continuous_Average_Price",
        # ---
        "Intraday Continuous (DE-AT-LU), volume":"ID_Continuous_Volume",
        "Intraday Continuous (DE-LU), volume":"ID_Continuous_Volume",
        "Intraday Continuous (DE-AT-LU, DE-LU), volume":"ID_Continuous_Volume",
        "Intraday Continuous  (DE-LU), volume":"ID_Continuous_Volume",
        "Intraday Continuous  (DE-AT-LU), volume": "ID_Continuous_Volume",
        "Intraday Continuous  (DE-AT-LU, DE-LU), volume": "ID_Continuous_Volume"
    }
    prepare_and_store_market(country, years, market, "intraday", mapping=mapping)

    # weather conditions
    mapping={}
    for quantity in [
        "wind_speed", "wind_direction", "air_temperature", "air_humidity", "solar_globe", "solar_diffuse"
    ]:
        prepare_and_store_weather(country,years,quantity=quantity,mapping=mapping)

    print("All data accuired")

if __name__ == '__main__':
    main()
