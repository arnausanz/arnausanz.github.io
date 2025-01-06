import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
import pandas as pd

"""
This file contains utility functions used in the project.
"""

def set_meteocat_api_key() -> str:
    """
    Returns the Meteocat API key stored in the .env file.
    :return: Meteocat API key
    """
    load_dotenv()
    return os.getenv("METEOCAT_API_KEY")

def save_df_to_csv(df, file_name, path = 'data/raw', header = True) -> None:
    """
    Saves a DataFrame to a CSV file using root dir to be able to update the data from GitHub Actions.
    :param df: DataFrame to save
    :param file_name: Name of the file
    :param path: Path to save the file (relative from the project root)
    :param header: Whether to include the header in the CSV file
    :return: None
    """
    project_dir = get_root_dir()
    _path = os.path.join(project_dir, path)
    df.to_csv(_path + file_name + '.csv', index=False, header=header)

def get_root_dir():
    """
    Returns the root directory of the project.
    :return: Root directory of the project
    """
    return os.path.dirname(os.path.dirname(__file__))

def get_date(date_str):
    """
    Returns the date in the format 'dd/mm/YYYY' for the given string ('yesterday', 'today').
    :param date_str: String with the date ('yesterday', 'today')
    :return: Date in the format 'dd/mm/YYYY'
    """
    if date_str == 'yesterday':
        return_date_value = datetime.now().date() - timedelta(days=1)
    elif date_str == 'today':
        return_date_value = datetime.now().date()
    else:
        return_date_value = None
    return return_date_value

def update_today_data_file(variables = ('aca', '1300', '1000', '1600')):
    """
    Updates the today data files for the given variables and prints a success message.
    :param variables: Tuple with the variables to update ('aca', '1300', '1000', '1600')
    :return: None
    """
    if 'aca' in variables:
        df = pd.read_csv(get_root_dir() + '/data/processed/aca/aca_daily_all.csv')
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
        df['date'] = df['date'].dt.date
        df = df[df['date'] == get_date('today')]
        save_df_to_csv(df, 'aca_daily_today', 'data/processed/aca/')
    if '1300' in variables:
        df = pd.read_csv(get_root_dir() + '/data/processed/meteocat/meteocat_1300_daily_all.csv')
        df['data'] = pd.to_datetime(df['data'], format='%Y-%m-%d')
        df['data'] = df['data'].dt.date
        df = df[df['data'] == get_date('today')]
        save_df_to_csv(df, 'meteocat_1300_daily_today', 'data/processed/meteocat/')
    if '1000' in variables:
        df = pd.read_csv(get_root_dir() + '/data/processed/meteocat/meteocat_1000_daily_all.csv')
        df['data'] = pd.to_datetime(df['data'], format='%Y-%m-%d')
        df['data'] = df['data'].dt.date
        df = df[df['data'] == get_date('today')]
        save_df_to_csv(df, 'meteocat_1000_daily_today', 'data/processed/meteocat/')
    if '1600' in variables:
        df = pd.read_csv(get_root_dir() + '/data/processed/meteocat/meteocat_1600_daily_all.csv')
        df['data'] = pd.to_datetime(df['data'], format='%Y-%m-%d')
        df['data'] = df['data'].dt.date
        df = df[df['data'] == get_date('today')]
        save_df_to_csv(df, 'meteocat_1600_daily_today', 'data/processed/meteocat/')
    print('Today data files updated successfully')

update_today_data_file()