import requests
import pandas as pd
from utils import calling_api_message, set_meteocat_api_key, save_df_to_csv

"""
This script is used to extract daily data from Meteocat API.
In the metadata files can be found the variable codes and station codes.
"""

__HEADERS = {"Content-Type": "application/json", "X-Api-Key": set_meteocat_api_key()}
__XEMA_BASE_URL = "https://api.meteo.cat/xema/v1/"
__DAILY_DATA_URL = [__XEMA_BASE_URL + "variables/estadistics/diaris/", "?codiEstacio=", "&any=", "&mes="]
__VARIABLE_FOR_ALL_STATIONS = ["variables/mesurades/{var_code}/{year}/{month}/{day}", "?codiEstacio={station_code}"]

def get_daily_data(var_code = None, station_code = None, year = None, month = None) -> pd.DataFrame:
    """
    OBSOLETE --> Won't use this function as we can get all the data from one variable from the API in one call
    """
    url = __DAILY_DATA_URL[0] + var_code + __DAILY_DATA_URL[1] + station_code + __DAILY_DATA_URL[2] + year + __DAILY_DATA_URL[3] + month
    response = requests.get(url, headers=__HEADERS)
    calling_api_message('Meteocat', 'Daily data from station: ' + station_code + ' - Year: ' + year + ' Month: ' + month + ' - Var: ' + var_code)
    print('Response status code: ' + str(response.status_code))
    data_df = pd.json_normalize(response.json(), record_path=['valors'], meta=['codiEstacio', 'codiVariable'])
    return data_df

def get_all_stations_data_per_day_and_var(var_code = None, year = None, month = None, day = None, station_code = None) -> pd.DataFrame:
    url = __XEMA_BASE_URL + __VARIABLE_FOR_ALL_STATIONS[0].format(var_code=var_code, year=year, month=month, day=day)
    url += __VARIABLE_FOR_ALL_STATIONS[1].format(station_code=station_code) if station_code is not None else ''
    response = requests.get(url, headers=__HEADERS)
    calling_api_message('Meteocat', 'Daily data from all stations - Year: ' + year + ' Month: ' + month + ' - Var: ' + var_code)
    print('Response status code: ' + str(response.status_code))
    data = pd.json_normalize(response.json(), record_path=['variables', 'lectures'], meta=['codi'])
    return data

save_df_to_csv(get_all_stations_data_per_day_and_var('35', '2024', '09', '21'), "test5")