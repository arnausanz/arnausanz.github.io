import pandas as pd
import numpy as np
from geopy.distance import geodesic
import DataExtraction.utils as utils

_var_code_paths = {
    'aca': utils.get_root_dir() + '/data/processed/aca/aca_daily_all.csv',
    '1300': utils.get_root_dir() + '/data/processed/meteocat/meteocat_1300_daily_all.csv',
    '1000': utils.get_root_dir() + '/data/processed/meteocat/meteocat_1000_daily_all.csv',
    '1600': utils.get_root_dir() + '/data/processed/meteocat/meteocat_1600_daily_all.csv'
}

def meteocat_data(var_code):
    data = pd.read_csv(_var_code_paths[var_code])
    data['data'] = pd.to_datetime(data['data'], format='%Y-%m-%d')
    return data

def aca_data():
    data = pd.read_csv(_var_code_paths['aca'])
    data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')
    return data

def icgc_data():
    data = pd.read_csv('data/processed/icgc/cobretes_sol.csv')
    return data

def read_metadata():
    meteocat_stations_metadata = pd.read_csv(utils.get_root_dir() + 'data/processed/meteocat/stations_metadata.csv')
    aca_sensors_metadata = pd.read_csv(utils.get_root_dir() + 'data/processed/aca/sensors_metadata.csv')
    meteocat_stations_metadata['coordenades.latitud'] = meteocat_stations_metadata['coordenades.latitud'].astype(np.float64)
    meteocat_stations_metadata['coordenades.longitud'] = meteocat_stations_metadata['coordenades.longitud'].astype(np.float64)
    aca_sensors_metadata['latitude'] = aca_sensors_metadata['latitude'].astype(np.float64)
    aca_sensors_metadata['longitude'] = aca_sensors_metadata['longitude'].astype(np.float64)
    return aca_sensors_metadata, meteocat_stations_metadata

meteocat_data_1000 = meteocat_data('1000')
meteocat_data_1300 = meteocat_data('1300')
meteocat_data_1600 = meteocat_data('1600')
aca_data = aca_data()
aca_sensors_metadata, meteocat_stations_metadata = read_metadata()

def calc_distances():
    """
    Calculate the distances between all the stations for each reservoir
    :return: pd.DataFrame
    """
    distances = []
    for _, aca_sensor in aca_sensors_metadata.iterrows():
        for _, meteocat_station in meteocat_stations_metadata.iterrows():
            distance = geodesic((aca_sensor['latitude'], aca_sensor['longitude']), (meteocat_station['coordenades.latitud'], meteocat_station['coordenades.longitud'])).km
            distances.append((aca_sensor['codiEstacio'], meteocat_station['codi'], distance))

    return pd.DataFrame(distances, columns=['aca_sensor', 'meteocat_station', 'distance'])


print(calc_distances())