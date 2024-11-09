# This document will be used to extract and update all the data from the different APIs. It will be called manually when needed.
import os
import time
from datetime import datetime
import pandas as pd
import meteocat
import utils

_APIs = ['meteocat_raw']
_METEOCAT_VAR_CODES = [1300, 1000, 1600]
_METEOCAT_START_YEAR = 2000 # As we only have data from 2000 by ACA
_METEOCAT_START_MONTH = 1

def update_all_data(api = None):
    pass


def get_all_meteocat_data(var_code = None, start_year = _METEOCAT_START_YEAR, start_month = _METEOCAT_START_MONTH, delay = 5, current_year = datetime.now().year, current_month = datetime.now().month):
    if start_year < _METEOCAT_START_YEAR:
        raise ValueError("Year must be greater than " + str(_METEOCAT_START_YEAR))
    if start_month < 1 or start_month > 12:
        raise ValueError("Month must be between 1 and 12")
    if var_code not in _METEOCAT_VAR_CODES and var_code is not None:
        raise ValueError("Variable code not valid")
    var_code = _METEOCAT_VAR_CODES if var_code is None else [var_code]

    for year in range(start_year, current_year + 1):
        for month in range(start_month, 13):
            if year == current_year and month > current_month:
                break
            if month < 10:
                month = "0" + str(month)
            for var in var_code:
                data = meteocat.get_daily_data(var, str(year), str(month), verbose=False)
                utils.save_df_to_csv(data, "meteocat_" + str(var) + "_" + str(year) + "_" + str(month), "data/raw/meteocat_raw/"+str(var)+"/")
        time.sleep(delay)
        start_month = 1

def concat_all_meteocat_data(varcode, folder_path = 'data/raw/meteocat_raw/'):
    previous_dir = os.getcwd()
    root_dir = utils.get_root_dir()
    os.chdir(root_dir + "/" + folder_path + varcode)
    files = os.listdir()
    df = pd.DataFrame()
    for f in files:
        if len(df) == 0:
            df = pd.read_csv(f)
        else:
            df = pd.concat([df, pd.read_csv(f)])
    os.chdir(previous_dir)
    return df

def transform_all_meteocat_data(data):
    # Ignore percentatge "column"
    data = data.drop(columns=['percentatge'])
    # Parse date
    data['data'] = data['data'].apply(utils.parse_date)
    # Sort by date and station code
    data = data.sort_values(by=['data', 'codiEstacio'])
    return data


def save_all_meteocat_data_to_csv(data, file_name, path = 'data/processed/meteocat/'):
    utils.save_df_to_csv(data, file_name, path = path)



# TODO: Implement update_all_data function
"""
CAUTION --> This snippet could exceed the API limit of requests. Use it carefully


get_all_meteocat_data(1000, current_year=2008, current_month=12)
a = concat_all_meteocat_data('1000')
save_all_meteocat_data_to_csv(transform_all_meteocat_data(a), file_name= 'meteocat_1000_daily_all')

get_all_meteocat_data(1600, start_year=2004, start_month=6, current_year=2008, current_month=12)
b = concat_all_meteocat_data('1600')
save_all_meteocat_data_to_csv(transform_all_meteocat_data(b), file_name= 'meteocat_1600_daily_all')
"""