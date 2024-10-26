import pandas as pd
import requests

from DataExtraction import utils

"""
This file is used to extract daily data from ACA API.
As real time data is only available for 3 months, the oldest data is downloaded from: https://analisi.transparenciacatalunya.cat/Medi-Ambient/Quantitat-d-aigua-als-embassaments-de-les-Conques-/gn9e-3qhr/about_data
"""

_CATALOG_BASE_URL = "http://aca-web.gencat.cat/sdim2/apirest/catalog?componentType="

def get_daily_data(date_from, dato_to):
    # a = requests.get("http://aca-web.gencat.cat/sdim2/apirest/data/EMBASSAMENT-EST/083036-001-ANA023?limit=7&from=01/10/2024T09:00:00&to=14/10/2024T12:00:00")
    # print(a.json())
    # TODO: Implement this function --> Must use the metadata to get the proper information of all water reservoirs
    pass


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

def transform_metadata(aca_metadata_src = "DataExtraction/metadata/aca_metadata/embassaments_metadata.csv", save_path = "data/processed/aca/", save_name = "geolocation_metadata"):
    aca_metadata_src = utils.get_root_dir() + "/" + aca_metadata_src
    df = pd.read_csv(aca_metadata_src, dtype={'location': 'str'})
    df = df[['componentDesc', 'location']]
    df['latitude'] = df['location'].apply(lambda x: x.split(" ")[0])
    df['longitude'] = df['location'].apply(lambda x: x.split(" ")[1])
    df.drop(columns=['location'], inplace=True)
    df.drop_duplicates(inplace=True)
    utils.save_df_to_csv(df, save_name, save_path)


# update_aca_metadata() --> To update embassaments_metadata.csv
# transform_metadata() --> To transform embassaments_metadata.csv
# transform_historical_data() --> To transform historical_data.csv