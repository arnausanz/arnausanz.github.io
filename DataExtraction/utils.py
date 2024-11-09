import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
import pandas as pd

def calling_api_message(api_name, more_info = None):
    print(f"Calling {api_name} API")
    if more_info is not None:
        print("Retrieving information: " + more_info)

def print_response_info(response):
    print('Response status code: ' + str(response.status_code) + " - " + response.reason)

def set_meteocat_api_key() -> str:
    load_dotenv()
    return os.getenv("METEOCAT_API_KEY_2")

def parse_date(date: str, input_format="%Y-%m-%dZ") -> datetime:
    return datetime.strptime(date, input_format)

def save_df_to_csv(df, file_name, path = 'data/raw', header = True) -> None:
    project_dir = get_root_dir()
    _path = os.path.join(project_dir, path)
    df.to_csv(_path + file_name + '.csv', index=False, header=header)

def check_str_int(value):
    print(value)
    if isinstance(value, int):
        return str(value)
    elif isinstance(value, str):
        return value
    else:
        raise ValueError("Value must be a string or an integer")

def get_root_dir():
    return os.path.dirname(os.path.dirname(__file__))

def get_date(date_str):
    if date_str == 'yesterday':
        return_date_value = datetime.now().date() - timedelta(days=1)
    elif date_str == 'today':
        return_date_value = datetime.now().date()
    else:
        return_date_value = None
    return return_date_value.strftime("%d/%m/%Y")

def update_today_data_file(variables = ('aca', '1300', '1000', '1600')):
    if 'aca' in variables:
        df = pd.read_csv(get_root_dir() + '/data/processed/aca/aca_daily_all.csv')
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
        df = df[df['date'] == get_date('today')]
        save_df_to_csv(df, 'aca_daily_today', 'data/processed/aca/')
    if '1300' in variables:
        df = pd.read_csv(get_root_dir() + '/data/processed/meteocat/meteocat_1300_daily_all.csv')
        df['data'] = pd.to_datetime(df['data'], format='%Y-%m-%d')
        df = df[df['data'] == get_date('today')]
        save_df_to_csv(df, 'meteocat_1300_daily_today', 'data/processed/meteocat/')
    if '1000' in variables:
        df = pd.read_csv(get_root_dir() + '/data/processed/meteocat/meteocat_1000_daily_all.csv')
        df['data'] = pd.to_datetime(df['data'], format='%Y-%m-%d')
        df = df[df['data'] == get_date('today')]
        save_df_to_csv(df, 'meteocat_1000_daily_today', 'data/processed/meteocat/')
    if '1600' in variables:
        df = pd.read_csv(get_root_dir() + '/data/processed/meteocat/meteocat_1600_daily_all.csv')
        df['data'] = pd.to_datetime(df['data'], format='%Y-%m-%d')
        df = df[df['data'] == get_date('today')]
        save_df_to_csv(df, 'meteocat_1600_daily_today', 'data/processed/meteocat/')

# update_today_data_file() # --> To get today data files used in visualization