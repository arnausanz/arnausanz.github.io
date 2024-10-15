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

# update_aca_metadata() --> To update embassaments_metadata.csv