import ast
import numpy as np
import requests
import pandas as pd
import DataExtraction.utils as utils
import datetime

"""
This file is used to extract daily data from meteocat_raw API.
In the metadata files can be found the variable codes and station codes.
"""

__HEADERS = {"Content-Type": "application/json", "X-Api-Key": utils.set_meteocat_api_key()}
__XEMA_BASE_URL = "https://api.meteo.cat/xema/v1/"
__DAILY_DATA_URL = [__XEMA_BASE_URL + "variables/estadistics/diaris/", "?codiEstacio=", "&any=", "&mes=", "?any=", "&mes="]
__MEASURED_DATA_URL = [__XEMA_BASE_URL + "variables/mesurades/", "/", "/", "/", "?codiEstacio="]

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

def add_today_information(var_code):
    # Find the first tuple value of the map when passing the second one
    corrected_var = ""
    for key, value in _MAP_DAILY_MEASURED_VARS.items():
        if value[1] == var_code:
            corrected_var = int(_MAP_DAILY_MEASURED_VARS[key][0])
            break
    year = str(datetime.datetime.now().year)
    month = str(datetime.datetime.now().month)
    day = str(datetime.datetime.now().day)
    url = __MEASURED_DATA_URL[0] + var_code + __MEASURED_DATA_URL[1] + year + __MEASURED_DATA_URL[2] + month + __MEASURED_DATA_URL[3] + day + __MEASURED_DATA_URL[4]
    response = requests.get(url, headers=__HEADERS)
    df = pd.json_normalize(response.json(), record_path=['variables', 'lectures'], meta=['codi', ['variables', 'codi']])
    df['data'] = df['data'].apply(lambda x: utils.parse_date(x, input_format="%Y-%m-%dT%H:%MZ"))
    df['data'] = df['data'].dt.date
    df.drop(columns=['estat', 'baseHoraria', 'variables.codi'], inplace=True)
    df_final_today = df.groupby(['data', 'codi']).sum().reset_index()
    df_final_today['codiVariable'] = corrected_var
    df_final_today.rename(columns={'codi': 'codiEstacio'}, inplace=True)
    return df_final_today



def transform_daily_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    This function transforms the daily data from the meteocat API.
    :param data: DataFrame with the daily data
    :return: DataFrame with the transformed data
    """
    data.drop(columns=['percentatge'], inplace=True)
    data['data'] = data['data'].apply(utils.parse_date)
    data['data'] = data['data'].dt.date
    return data

def join_daily_and_today_data(daily_tf, today):
    return pd.concat([daily_tf, today])

def data_getter(var_name):
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
    # It can be yet duplicated data from some variables
    # Analyze if existing data contains a row for the same date and station as any in the new_data and drop it before concatenating
    existing_data = existing_data[~existing_data.apply(lambda x: (x['data'], x['codiEstacio']) in new_data[['data', 'codiEstacio']].apply(tuple, axis=1), axis=1)]
    df = pd.concat([existing_data, new_data])
    df.sort_values(by=['data', 'codiEstacio'], inplace=True, ascending=False, ignore_index=True)
    df.reset_index(drop=True, inplace=True)
    df.drop_duplicates(inplace=True)
    if overwrite:
        utils.save_df_to_csv(df, f"meteocat_{new_data['codiVariable'].iloc[0]}_daily_all", path="data/processed/meteocat/")
    else:
        return df

def print_api_usage():
    print(requests.get('https://api.meteo.cat/quotes/v1/consum-actual', headers=__HEADERS).json())


def concat_meteocat_data_from_two_different_sources(data_from_manual_source_path, data_from_api_source_path):
    """
    # TODO -> Aquesta funció serveix per unir les variables 1000 i 1600 amb les corresponents extretes manualment (32 i 38)
    :param data_from_api_source_path:
    :param data_from_manual_source_path:
    :return: DataFrame with concatenated data
    """
    manual_data_df = pd.read_csv(data_from_manual_source_path)
    api_data_df = pd.read_csv(data_from_api_source_path)
    manual_data_df['by_day_data_lectura'] = manual_data_df['by_day_data_lectura'].apply(lambda x: utils.parse_date(x, input_format='%m/%d/%Y %I:%M:%S %p'))
    manual_data_df['CODI_VARIABLE'] = api_data_df['codiVariable'].iloc[0]
    manual_data_df['VALOR_LECTURA'] = manual_data_df['VALOR_LECTURA'].astype(float)
    manual_data_df.sort_values(by=['by_day_data_lectura'], inplace=True, ascending=False, ignore_index=True)
    manual_data_df = manual_data_df[['by_day_data_lectura', 'VALOR_LECTURA', 'CODI_ESTACIO', 'CODI_VARIABLE']].copy()
    manual_data_df.columns = api_data_df.columns
    api_data_df['data'] = pd.to_datetime(api_data_df['data'], format='%Y-%m-%d')
    result = pd.concat([api_data_df, manual_data_df])
    result.sort_values(by=['data'], inplace=True, ascending=False, ignore_index=True)
    return result

def log_meteocat_data(var_name, year, month, trigger = "Manual"):
    log_file = 'logs/meteocat_data_update.txt'
    msg = f"Var: {var_name} - Year: {year} - Month: {month}"
    with open(utils.get_root_dir() + '/' + log_file, 'a') as f:
        f.write(f"{datetime.datetime.strftime(datetime.datetime.now(), '%d-%m-%Y %H:%M:%S')} - Updated by: {trigger} - {msg}\n")
        f.close()

def get_stations_metadata():
    """
    Function to get the metadata with coordinates of the stations (only the active ones)
    :return:
    """
    path = utils.get_root_dir() + '/DataExtraction/metadata/meteocat_metadata/stations_metadata.csv'
    data = pd.read_csv(path)
    data['estats'] = data['estats'].apply(lambda x: ast.literal_eval(x))
    data['estats'] = data['estats'].apply(lambda x: x if len(x) == 1 else np.nan)
    data.dropna(subset=['estats'], inplace=True)
    data[['codi', 'altitud', 'coordenades.latitud', 'coordenades.longitud']].to_csv(utils.get_root_dir() + '/data/processed/meteocat/stations_metadata.csv', index=False)


# get_stations_metadata()

# save_df_to_csv(get_daily_data("1300",  "1989", "02"), "test7")

# join_meteocat_data(data_getter("1300"), transform_daily_data(get_daily_data("1300",  "2024", "10")), overwrite=True)

# joined_32_1000 = concat_meteocat_data_from_tow_different_sources('../data/raw/meteocat_raw/32/32_manual.csv', '../data/processed/meteocat/meteocat_1000_daily_all.csv')
# joined_38_1600 = concat_meteocat_data_from_tow_different_sources('../data/raw/meteocat_raw/38/38_manual.csv', '../data/processed/meteocat/meteocat_1600_daily_all.csv')
# utils.save_df_to_csv(joined_32_1000, 'meteocat_1000_daily_all', 'data/processed/meteocat/')
# utils.save_df_to_csv(joined_38_1600, 'meteocat_1600_daily_all', 'data/processed/meteocat/')