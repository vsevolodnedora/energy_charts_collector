Fetch historical data related to electricity market from https://www.energy-charts.info/
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
  - `DE_weather_air_humidity_2015_2024` air humidity
  - `DE_weather_air_temperature_2015_2024` air temperature
  - `DE_weather_air_solar_diffuse_2015_2024` solar diffuse
  - `DE_weather_air_solar_globe_2015_2024` solar globe
  - `DE_weather_air_wind_direction_2015_2024` wind direction
  - `DE_weather_air_wind_speed_2015_2024` wind speed

The weather data is provided for all available weather stations (see weather_stations.csv for
their coordinates and elevation)

_NOTE:_ due to licencing data itself cannot be provided here.
Thus at least one run of the script is required to fetch it.

_NOTE:_ because yearly 'highcharts' can be very busy, loading takes time and
sometimes fails. Several methods to overcome it are implemented. However, sometimes,
re-running the script helps (it will skip over already downloaded and cached data)

_NOTE:_ over the years trading zones have changed. DE-AT-LU became DE-LU and then DE in some
markets. We simplify it all here by removing them everywhere using mapping.

_NOTE:_ Power data is given with 15min frequency and thus might require downsamling to be
  aligned with the rest. Also, no usual aggregation is done for power generation.

_NOTE:_ climate data is scraped differently from the others as it is not available for the
whole year for hourly intervals. Thus, the code scrape it for each month of a given year
and then concatenate the result. Also note, that the climate data is available differently
for different cities in Germany and only the data that has number of entries expected for
the hourly data is stored.