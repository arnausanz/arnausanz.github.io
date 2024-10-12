import os
from ast import parse
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

def calling_api_message(api_name, more_info = None):
    print(f"Calling {api_name} API")
    if more_info is not None:
        print("Retrieving information: " + more_info)

def set_meteocat_api_key() -> str:
    load_dotenv()
    return os.getenv("METEOCAT_API_KEY")

def parse_date(date: str) -> datetime:
    return datetime.strptime(date, '%Y-%m-%dZ')

def save_df_to_csv(df, file_name, path = 'Data/') -> None:
    project_dir = os.path.dirname(os.path.dirname(__file__))
    _path = os.path.join(project_dir, path)
    df.to_csv(_path + file_name, index=False)