import requests
import pandas as pd
from utils import calling_api_message, set_meteocat_api_key

__HEADERS = {"Content-Type": "application/json", "X-Api-Key": "__KEY__"}
__XEMA_BASE_URL = "https://api.meteo.cat/xema/v1/"
__DAILY_DATA_URL = [__XEMA_BASE_URL + "variables/estadistics/diaris/", "?codiEstacio=", "&any=", "&mes="]

def get_daily_data(var_code = None, station_code = None, year = None, month = None) -> pd.DataFrame:
    url = __DAILY_DATA_URL[0] + var_code + __DAILY_DATA_URL[1] + station_code + __DAILY_DATA_URL[2] + year + __DAILY_DATA_URL[3] + month
    __HEADERS['X-Api-Key'] = set_meteocat_api_key()
    response = requests.get(url, headers=__HEADERS)
    calling_api_message('Meteocat', 'Daily data from station: ' + station_code + ' - Year: ' + year + ' Month: ' + month + ' - Var: ' + var_code)
    print('Response status code: ' + str(response.status_code))
    data_df = pd.json_normalize(response.json(), record_path=['valors'], meta=['codiEstacio', 'codiVariable'])
    return data_df

print(get_daily_data('1300', 'UG', '2024', '01'))