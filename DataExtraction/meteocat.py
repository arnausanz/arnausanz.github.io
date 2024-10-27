import requests
import pandas as pd
import utils

"""
This file is used to extract daily data from meteocat_raw API.
In the metadata files can be found the variable codes and station codes.
"""

__HEADERS = {"Content-Type": "application/json", "X-Api-Key": utils.set_meteocat_api_key()}
__XEMA_BASE_URL = "https://api.meteo.cat/xema/v1/"
__DAILY_DATA_URL = [__XEMA_BASE_URL + "variables/estadistics/diaris/", "?codiEstacio=", "&any=", "&mes=", "?any=", "&mes="]
__VARIABLE_FOR_ALL_STATIONS = ["variables/mesurades/{var_code}/{year}/{month}/{day}", "?codiEstacio={station_code}"]

def get_daily_data(var_code = None, year = None, month = None, station_code = None, verbose=True) -> pd.DataFrame:
    var_code, year, month = utils.check_str_int(var_code), utils.check_str_int(year), utils.check_str_int(month)
    station_code = "" if station_code is None else station_code
    url = __DAILY_DATA_URL[0] + var_code + __DAILY_DATA_URL[1] + station_code + __DAILY_DATA_URL[2] + year + __DAILY_DATA_URL[3] + month
    response = requests.get(url, headers=__HEADERS)
    if station_code == "" and verbose:
        utils.calling_api_message('meteocat_raw', 'Daily data from all stations - Year: ' + year + ' Month: ' + month + ' - Var: ' + var_code)
    elif verbose:
        utils.calling_api_message('meteocat_raw', 'Daily data from station: ' + station_code + ' - Year: ' + year + ' Month: ' + month + ' - Var: ' + var_code)
    utils.print_response_info(response) if verbose else None
    data_df = pd.json_normalize(response.json(), record_path=['valors'], meta=['codiEstacio', 'codiVariable'])
    return data_df


# save_df_to_csv(get_daily_data("1300",  "1989", "02"), "test7")