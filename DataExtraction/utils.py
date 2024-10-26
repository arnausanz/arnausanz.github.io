import os
from ast import parse
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

def calling_api_message(api_name, more_info = None):
    print(f"Calling {api_name} API")
    if more_info is not None:
        print("Retrieving information: " + more_info)

def print_response_info(response):
    print('Response status code: ' + str(response.status_code) + " - " + response.reason)

def set_meteocat_api_key() -> str:
    load_dotenv()
    return os.getenv("METEOCAT_API_KEY")

def parse_date(date: str, input_format="%Y-%m-%dZ") -> datetime:
    return datetime.strptime(date, input_format)

def save_df_to_csv(df, file_name, path = 'data/raw', header = True) -> None:
    project_dir = get_root_dir()
    _path = os.path.join(project_dir, path)
    df.to_csv(_path + file_name + '.csv', index=False, header=header)

def save_df_to_json(data, file_name, path = 'data/raw') -> None:
    project_dir = get_root_dir()
    _path = os.path.join(project_dir, path)
    with open(_path + file_name + '.json', 'w') as f:
        f.write(str(data))

def check_str_int(value):
    if isinstance(value, int):
        return str(value)
    elif isinstance(value, str):
        return value
    else:
        raise ValueError("Value must be a string or an integer")

def get_root_dir():
    return os.path.dirname(os.path.dirname(__file__))