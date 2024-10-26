import pandas as pd
import requests

from DataExtraction import utils

"""
This file is used to extract daily data from ACA API.
As real time data is only available for 3 months, the oldest data is downloaded from: https://analisi.transparenciacatalunya.cat/Medi-Ambient/Quantitat-d-aigua-als-embassaments-de-les-Conques-/gn9e-3qhr/about_data
"""

_CATALOG_BASE_URL = "http://aca-web.gencat.cat/sdim2/apirest/catalog?componentType="
_EMBASSAMENT_DATA_BASE_URL = ["http://aca-web.gencat.cat/sdim2/apirest/data/EMBASSAMENT-EST/", "?limit=-1&from=", "T00:00:00&to=", "T23:59:59"]

_NAME_MATCHING_DICT = {
    'Foix (Castellet i la Gornal)': 'Embassament de Foix (Castellet i la Gornal)',
    'Darnius Boadella (Darnius)': 'Embassament de Darnius Boadella (Darnius)',
    'Gaià (el Catllar)': 'Embassament de Gaià (el Catllar)',
    'Riudecanyes' : 'Embassament de Riudecanyes',
    'Siurana (Cornudella de Montsant)': 'Embassament de Siurana (Cornudella de Montsant)',
    'Pasteral (la Cellera de Ter)': 'Embassament de Pasteral (la Cellera de Ter)',
    'Sau (Vilanova de Sau)': 'Embassament de Sau (Vilanova de Sau)',
    'Llosa del Cavall (Navès)': 'Embassament de la Llosa del Cavall (Navès)',
    'Baells (Cercs)': 'Embassament de la Baells (Cercs)',
    'Sant Ponç (Clariana de Cardener)': 'Embassament de Sant Ponç (Clariana de Cardener)',
    'Susqueda (Osor)': 'Embassament de Susqueda (Osor)'
}

def get_all_sensor_codes(sensor_data_file = 'data/processed/aca/sensor_metadata.csv') -> list:
    df = pd.read_csv(utils.get_root_dir() + '/' + sensor_data_file)
    return df['sensor'].tolist()

def get_sensor_name_relation(sensor_data_file = 'data/processed/aca/sensor_metadata.csv') -> pd.DataFrame:
    df = pd.read_csv(utils.get_root_dir() + '/' + sensor_data_file)
    return df[['sensor', 'componentDesc']]


def get_daily_data(date_from, date_to, sensor_code = 'all'):
    # If dates are None, it takes date_from as yesterday and date_to as today
    if date_from is None:
        date_from = utils.get_date('yesterday')
    if date_to is None:
        date_to = utils.get_date('today')

    if sensor_code == 'all':
        sensor_codes = get_all_sensor_codes()
    else:
        sensor_codes = [sensor_code] if type(sensor_code) == str else sensor_code
    for sensor in sensor_codes:
        url = _EMBASSAMENT_DATA_BASE_URL[0] + sensor + _EMBASSAMENT_DATA_BASE_URL[1] + date_from + _EMBASSAMENT_DATA_BASE_URL[2] + date_to + _EMBASSAMENT_DATA_BASE_URL[3]
        response = requests.get(url)
        df = pd.json_normalize(response.json(), record_path=['observations'])
        df['sensor'] = sensor
        print(df)

# TODO --> S'ha de fer que el get_daily_data retorni un dataframe amb totes les dades i que es guardi fusioni amb el aca_daily_all.csv
get_daily_data(None, None, 'CALC000004')

def get_catalog(catalog_type="embassament"):
    url = _CATALOG_BASE_URL + catalog_type
    response = requests.get(url)
    return pd.json_normalize(response.json(), record_path=['providers', 'sensors'])

def update_aca_metadata():
    utils.save_df_to_csv(get_catalog(), "embassaments_metadata", "DataExtraction/metadata/aca_metadata/")

def transform_historical_data(aca_historical_data_src = "data/raw/aca/manual_download/historical_data.csv", save_path = "data/processed/aca/", save_name = "aca_daily_all"):
    aca_historical_data_src = utils.get_root_dir() + "/" + aca_historical_data_src
    df = pd.read_csv(aca_historical_data_src, dtype={'Dia': 'str'})
    df['Dia'] = df['Dia'].apply(lambda x: utils.parse_date(x, "%d/%m/%Y"))
    df.sort_values(by=['Dia'], ascending=True, inplace=True)
    utils.save_df_to_csv(df, save_name, save_path)

def transform_metadata(aca_metadata_src = "DataExtraction/metadata/aca_metadata/embassaments_metadata.csv", save_path = "data/processed/aca/", save_name = "sensor_metadata"):
    aca_metadata_src = utils.get_root_dir() + "/" + aca_metadata_src
    df = pd.read_csv(aca_metadata_src, dtype={'location': 'str'})
    df = df[df['description']=='Volum embassat'][['componentDesc', 'location', 'sensor']]
    df['latitude'] = df['location'].apply(lambda x: x.split(" ")[0])
    df['longitude'] = df['location'].apply(lambda x: x.split(" ")[1])
    df.drop(columns=['location'], inplace=True)
    df.drop_duplicates(inplace=True)
    df.columns = ['name', 'sensor', 'latitude', 'longitude']
    df['name'] = df['name'].apply(lambda x: x.strip()).map(_NAME_MATCHING_DICT)
    utils.save_df_to_csv(df, save_name, save_path)


# update_aca_metadata() # --> To update embassaments_metadata.csv
# transform_metadata() # --> To transform embassaments_metadata.csv
# transform_historical_data() # --> To transform historical_data.csv