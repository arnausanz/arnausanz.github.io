import ast
import numpy as np
import requests
import pandas as pd
import utils
import datetime

"""
This file is used to extract daily data from meteocat_raw API.
In the metadata files can be found the variable codes and station codes.
"""

__HEADERS = {"Content-Type": "application/json", "X-Api-Key": utils.set_meteocat_api_key()}

_MAP_DAILY_MEASURED_VARS = {
    'precipitacio': ('1300', '35'),
    'temperatura': ('1000', '32'),
    'gruix_neu': ('1600', '38')
}

def get_daily_data(var_code = None, year = None, month = None) -> pd.DataFrame:
    """
    Retrieves the daily data from the meteocat API.
    :param var_code: code of the variable to retrieve
    :param year: Year of the data
    :param month: Month of the data
    :return: DataFrame with the daily data
    """
    url = "https://api.meteo.cat/xema/v1/variables/estadistics/diaris/" + var_code + "?codiEstacio=&any=" + "2025" + "&mes=" + "01"
    response = requests.get(url, headers=__HEADERS)
    data_df = pd.json_normalize(response.json(), record_path=['valors'], meta=['codiEstacio', 'codiVariable'])
    return data_df

def add_today_information(var_code):
    """
    Retrieve the information of the current day for the variable passed as parameter as the daily data with month and year doesn't provide it.
    As the variable code is different in the daily data and the today data, it is necessary to correct it.
    :param var_code: Variable code to retrieve the data (35, 32, 38)
    :return: DataFrame with the data of the current day and the corrected variable code
    """
    # Map the variable code to the correct one
    corrected_var = next((int(value[0]) for key, value in _MAP_DAILY_MEASURED_VARS.items() if value[1] == var_code), "")
    year = str(datetime.datetime.now().year)
    month = str(datetime.datetime.now().month) if len(str(datetime.datetime.now().month)) == 2 else "0" + str(datetime.datetime.now().month)
    day = str(datetime.datetime.now().day) if len(str(datetime.datetime.now().day)) == 2 else "0" + str(datetime.datetime.now().day)
    url = "https://api.meteo.cat/xema/v1/variables/mesurades/" + var_code + "/" + year + "/" + month + "/" + day + "?codiEstacio="
    response = requests.get(url, headers=__HEADERS)
    df = pd.json_normalize(response.json(), record_path=['variables', 'lectures'], meta=['codi', ['variables', 'codi']])
    df['data'] = pd.to_datetime(df['data'], format="%Y-%m-%dT%H:%MZ").dt.date
    df.drop(columns=['estat', 'baseHoraria', 'variables.codi'], inplace=True)
    # If variable is 35: sum (rain), if 32: mean (temperature), if 38: last value (snow accumulation)
    aggregation_functions = {
        '35': 'sum',
        '32': 'mean',
        '38': 'last'
    }
    if var_code in aggregation_functions:
        df_final_today = df.groupby(['data', 'codi']).agg(aggregation_functions[var_code]).reset_index()
    else:
        raise ValueError("Variable code not recognized")
    df_final_today['codiVariable'] = corrected_var
    df_final_today.rename(columns={'codi': 'codiEstacio'}, inplace=True)
    return df_final_today

def transform_daily_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms the daily data from the meteocat API.
    :param data: DataFrame with the daily data
    :return: DataFrame with the transformed data
    """
    data.drop(columns=['percentatge'], inplace=True) if 'percentatge' in data.columns else None
    data['data'] = pd.to_datetime(data['data'], format="%Y-%m-%dZ").dt.date
    return data

def join_daily_and_today_data(daily_tf, today):
    """
    Joins the daily data with the data of the current day.
    :param daily_tf: DataFrame with the daily data
    :param today: DataFrame with the data of the current day
    :return: DataFrame with the joined data
    """
    # Delete the data of the current day from the daily data if exists
    daily_tf['data'] = pd.to_datetime(daily_tf['data'], format='%Y-%m-%d')
    today['data'] = pd.to_datetime(today['data'], format='%Y-%m-%d')
    day = today['data'].unique()
    daily_tf = daily_tf[~daily_tf['data'].isin(day)]
    return pd.concat([daily_tf, today])

def data_getter(var_name):
    """
    Retrieves already stored data
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
    :return: DataFrame with the joined data if overwrite is False
    """
    existing_data['data'] = pd.to_datetime(existing_data['data'], format='%Y-%m-%d')
    new_data['data'] = pd.to_datetime(new_data['data'], format='%Y-%m-%d')
    # Drop data from same date, variable and station as in new data from existing data
    days = new_data['data'].unique()
    existing_data = existing_data[~existing_data['data'].isin(days)].copy()
    df = pd.concat([existing_data, new_data])
    df.sort_values(by=['data', 'codiEstacio'], inplace=True, ascending=False, ignore_index=True)
    df.reset_index(drop=True, inplace=True)
    df.drop_duplicates(inplace=True)
    if overwrite:
        utils.save_df_to_csv(df, f"meteocat_{new_data['codiVariable'].iloc[0]}_daily_all", path="data/processed/meteocat/")
    else:
        return df

def print_api_usage():
    """
    To check if we can exceed the limit of the API
    :return: None
    """
    print(requests.get('https://api.meteo.cat/quotes/v1/consum-actual', headers=__HEADERS).json())

"""
def concat_meteocat_data_from_two_different_sources(data_from_manual_source_path, data_from_api_source_path):
    #########################################################################################################
    As API usage was exceeded once, this function is used to concatenate the data from the API and the manual data extrected to avoid exceeding the limit again.
    -------------------------------------------
    AVOID USING THIS FUNCTION UNLESS NECESSARY
    -------------------------------------------
    :param data_from_api_source_path: Path to the data from the API
    :param data_from_manual_source_path: Path to the data from the manual extraction
    :return: DataFrame with concatenated data
    #########################################################################################################
    manual_data_df = pd.read_csv(data_from_manual_source_path)
    api_data_df = pd.read_csv(data_from_api_source_path)
    manual_data_df['by_day_data_lectura'] = manual_data_df['by_day_data_lectura'].apply(lambda x: datetime.datetime.strptime(x, '%m/%d/%Y %I:%M:%S %p'))
    manual_data_df['CODI_VARIABLE'] = api_data_df['codiVariable'].iloc[0]
    manual_data_df['VALOR_LECTURA'] = manual_data_df['VALOR_LECTURA'].astype(float)
    manual_data_df.sort_values(by=['by_day_data_lectura'], inplace=True, ascending=False, ignore_index=True)
    manual_data_df = manual_data_df[['by_day_data_lectura', 'VALOR_LECTURA', 'CODI_ESTACIO', 'CODI_VARIABLE']].copy()
    manual_data_df.columns = api_data_df.columns
    api_data_df['data'] = pd.to_datetime(api_data_df['data'], format='%Y-%m-%d')
    result = pd.concat([api_data_df, manual_data_df])
    result.sort_values(by=['data'], inplace=True, ascending=False, ignore_index=True)
    return result
"""

def log_meteocat_data(var_name, year, month, trigger = "Manual"):
    log_file = 'logs/meteocat_data_update.txt'
    msg = f"Var: {var_name} - Year: {year} - Month: {month}"
    with open(utils.get_root_dir() + '/' + log_file, 'a') as f:
        f.write(f"{datetime.datetime.strftime(datetime.datetime.now(), '%d-%m-%Y %H:%M:%S')} - Updated by: {trigger} - {msg}\n")
        f.close()

def get_stations_metadata():
    """
    To get the metadata with coordinates of the stations (only the active ones) and save it to a csv file.
    CAUTION --> It will overwrite the existing file.
    :return: None
    """
    path = utils.get_root_dir() + '/DataExtraction/metadata/meteocat_metadata/stations_metadata.csv'
    data = pd.read_csv(path)
    data['estats'] = data['estats'].apply(lambda x: ast.literal_eval(x))
    data['estats'] = data['estats'].apply(lambda x: x if len(x) == 1 else np.nan)
    data.dropna(subset=['estats'], inplace=True)
    data[['codi', 'altitud', 'coordenades.latitud', 'coordenades.longitud']].to_csv(utils.get_root_dir() + '/data/processed/meteocat/stations_metadata.csv', index=False)