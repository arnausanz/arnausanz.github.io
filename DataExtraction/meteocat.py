import requests
import pandas as pd
from utils import calling_api_message, set_meteocat_api_key, save_df_to_csv, check_str_int, save_df_to_json, print_response_info

"""
This file is used to extract daily data from meteocat_raw API.
In the metadata files can be found the variable codes and station codes.
"""

__HEADERS = {"Content-Type": "application/json", "X-Api-Key": set_meteocat_api_key()}
__XEMA_BASE_URL = "https://api.meteo.cat/xema/v1/"
__DAILY_DATA_URL = [__XEMA_BASE_URL + "variables/estadistics/diaris/", "?codiEstacio=", "&any=", "&mes=", "?any=", "&mes="]
__VARIABLE_FOR_ALL_STATIONS = ["variables/mesurades/{var_code}/{year}/{month}/{day}", "?codiEstacio={station_code}"]

def get_daily_data(var_code = None, year = None, month = None, station_code = None, verbose=True) -> pd.DataFrame:
    var_code, year, month = check_str_int(var_code), check_str_int(year), check_str_int(month)
    station_code = "" if station_code is None else station_code
    url = __DAILY_DATA_URL[0] + var_code + __DAILY_DATA_URL[1] + station_code + __DAILY_DATA_URL[2] + year + __DAILY_DATA_URL[3] + month
    response = requests.get(url, headers=__HEADERS)
    if station_code == "" and verbose:
        calling_api_message('meteocat_raw', 'Daily data from all stations - Year: ' + year + ' Month: ' + month + ' - Var: ' + var_code)
    elif verbose:
        calling_api_message('meteocat_raw', 'Daily data from station: ' + station_code + ' - Year: ' + year + ' Month: ' + month + ' - Var: ' + var_code)
    print_response_info(response) if verbose else None
    data_df = pd.json_normalize(response.json(), record_path=['valors'], meta=['codiEstacio', 'codiVariable'])
    return data_df

"""

DEPRECATED FUNCTION

def get_all_stations_data_per_day_and_var(var_code = None, year = None, month = None, day = None, station_code = None) -> pd.DataFrame:
    url = __XEMA_BASE_URL + __VARIABLE_FOR_ALL_STATIONS[0].format(var_code=var_code, year=year, month=month, day=day)
    url += __VARIABLE_FOR_ALL_STATIONS[1].format(station_code=station_code) if station_code is not None else ''
    response = requests.get(url, headers=__HEADERS)
    calling_api_message('meteocat_raw', 'Daily data from all stations - Year: ' + year + ' Month: ' + month + ' - Var: ' + var_code)
    print_response_info(response)
    data = pd.json_normalize(response.json(), record_path=['variables', 'lectures'], meta=['codi'])
    return data
"""

# save_df_to_csv(get_daily_data("1300",  "1989", "02"), "test7")