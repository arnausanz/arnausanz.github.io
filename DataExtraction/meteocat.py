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
__MEASURED_DATA_URL = ["variables/mesurades/{var_code}/{year}/{month}/{day}", "?codiEstacio={station_code}"]

_MAP_DAILY_MEASURED_VARS = {
    'precipitacio': ('1300', '35'),
    'temperatura': ('1000', '32'),
    'gruix_neu': ('1600', '38')
}

def get_daily_data(var_code = None, year = None, month = None, station_code = None, verbose=True) -> pd.DataFrame:
    """
    This function retrieves the daily data from the meteocat API.
    :param var_code: code of the variable to retrieve
    :param year: Year of the data
    :param month: Month of the data
    :param station_code: Code of the station to retrieve the data from, if None, all stations will be retrieved
    :param verbose: If True, it will print the API call message
    :return: DataFrame with the daily data
    """
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

def transform_daily_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    This function transforms the daily data from the meteocat API.
    :param data: DataFrame with the daily data
    :return: DataFrame with the transformed data
    """
    data.drop(columns=['percentatge'], inplace=True)
    data['data'] = data['data'].apply(utils.parse_date)
    return data

def _data_getter(var_name):
    """
    Function to retrieve already stored data
    :param var_name: Name of the variable
    :return: DataFrame with the data
    """
    return pd.read_csv(f"{utils.get_root_dir()}/data/processed/meteocat/meteocat_{var_name}_daily_all.csv")

def join_meteocat_data(existing_data, new_data, overwrite=True):
    """
    This function joins the existing meteocat data with the new data.
    :param existing_data: Existing data extracted from the file
    :param new_data: DataFrame with the new data
    :param overwrite: If True, it will overwrite the existing file
    :return: DataFrame with the joined data
    """
    existing_data['data'] = pd.to_datetime(existing_data['data'], format='%Y-%m-%d')
    new_data['data'] = pd.to_datetime(new_data['data'], format='%Y-%m-%d')
    df = pd.concat([existing_data, new_data])
    df.sort_values(by=['data', 'codiEstacio'], inplace=True, ascending=False, ignore_index=True)
    df.reset_index(drop=True, inplace=True)
    df.drop_duplicates(inplace=True)
    if overwrite:
        utils.save_df_to_csv(df, f"meteocat_{new_data['codiVariable'].iloc[0]}_daily_all", path="data/processed/meteocat/")

def print_api_usage():
    print(requests.get('https://api.meteo.cat/quotes/v1/consum-actual', headers=__HEADERS).json())

# save_df_to_csv(get_daily_data("1300",  "1989", "02"), "test7")

# join_meteocat_data(_data_getter("1300"), transform_daily_data(get_daily_data("1300",  "2024", "10")), overwrite=True)

# TODO --> implementar la funció que uneix les variables manuals (32 i 38) amb les automàtiques (1000 i 1600) i ajunar-ho al fitxer daily corresponent