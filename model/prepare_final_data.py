import numpy as np
from geopy.distance import geodesic
import geopandas as gpd
from scipy.spatial import cKDTree
from shapely.geometry.linestring import LineString
from tqdm import tqdm
from DataExtraction import utils
import pandas as pd

_var_code_paths = {
    'aca': utils.get_root_dir() + '/data/processed/aca/aca_daily_all.csv',
    '1300': utils.get_root_dir() + '/data/processed/meteocat/meteocat_1300_daily_all.csv',
    '1000': utils.get_root_dir() + '/data/processed/meteocat/meteocat_1000_daily_all.csv',
    '1600': utils.get_root_dir() + '/data/processed/meteocat/meteocat_1600_daily_all.csv',
    'icgc': utils.get_root_dir() + '/data/processed/icgc/cobertes_sol.csv',
    'aca_metadata': utils.get_root_dir() + '/data/processed/aca/sensor_metadata.csv',
    'meteocat_metadata': utils.get_root_dir() + '/DataExtraction/metadata/meteocat_metadata/stations_metadata.csv'
}

final_data_dir_path = utils.get_root_dir() + '/model/final_data/'

def prepare_aca_data(save = True):
    """
    Prepares the final ACA data for the model.
    :return: DataFrame with the final ACA data
    """
    print('Preparing ACA data...')
    processed_aca = pd.read_csv(_var_code_paths['aca'])
    processed_aca['date'] = pd.to_datetime(processed_aca['date'], format='%Y-%m-%d')
    processed_aca.sort_values(by='date', inplace=True)
    # Keep only date, name and value columns
    processed_aca = processed_aca[['date', 'name', 'current_volume']]
    # Ignore sensors with almost no data
    _ignored_sensors = ['Embassament de Pasteral (la Cellera de Ter)', 'Embassament de Gaià (el Catllar)']
    processed_aca = processed_aca[~processed_aca['name'].isin(_ignored_sensors)]
    if save:
        processed_aca.to_csv(final_data_dir_path + 'processed_aca.csv', index=False)
    return processed_aca

def prepare_icgc_data(save = True):
    """
    Prepares the final ICGC data for the model.
    :return: DataFrame with the final ICGC data
    """
    print('Preparing ICGC data...')
    # Get raw information with class and location
    processed_icgc = pd.read_csv(_var_code_paths['icgc'])
    meteocat_metadata = pd.read_csv(_var_code_paths['meteocat_metadata'])
    aca_metadata = pd.read_csv(_var_code_paths['aca_metadata'])[['name', 'latitude', 'longitude']].drop_duplicates()
    # Get geodataframes of icgc, meteo and aca
    crs = 'EPSG:4326'
    _geo_stations = gpd.GeoDataFrame(meteocat_metadata, geometry=gpd.points_from_xy(meteocat_metadata['coordenades.longitud'],meteocat_metadata['coordenades.latitud']), crs=crs).to_crs(epsg=3857)
    _geo_sensors = gpd.GeoDataFrame(aca_metadata, geometry=gpd.points_from_xy(aca_metadata['longitude'], aca_metadata['latitude']), crs=crs).to_crs(epsg=3857)
    _geo_soil = gpd.GeoDataFrame(processed_icgc, geometry=gpd.points_from_xy(processed_icgc['Longitude'], processed_icgc['Latitude']), crs=crs).to_crs(epsg=3857)
    soil_tree = cKDTree(np.vstack([_geo_soil.geometry.x, _geo_soil.geometry.y]).T)
    soil_info = []
    # Use a progress bar to track the processing
    with tqdm(total=_geo_sensors.shape[0]*_geo_stations.shape[0], desc="Processing sensors") as pbar:
        for _, sensor in _geo_sensors.iterrows():
            for _, station in _geo_stations.iterrows():
                # Get the line between the sensor and the station
                line = LineString([sensor['geometry'], station['geometry']])
                # Calculate the distance between the sensor and the station
                distance = geodesic((sensor['latitude'], sensor['longitude']),
                                    (station['coordenades.latitud'], station['coordenades.longitud'])).km
                # Interpolate points along the line
                num_points = int(distance * 10) # Calculate 10 points per km
                if num_points == 0:
                    continue # Ignore if num_points is 0
                interpolated_points = [line.interpolate(i / num_points, normalized=True) for i in range(num_points + 1)]
                # Find the nearest soil type for each interpolated point
                soil_types = []
                for point in interpolated_points:
                    point_coords = np.array([point.x, point.y])
                    _, idx = soil_tree.query(point_coords)
                    soil_types.append(_geo_soil.iloc[idx]['Class'])
                # Calculate the percentage of each soil type
                soil_type_counts = pd.Series(soil_types).value_counts(normalize=True) * 100
                # Initialize the soil type percentages with 0 for all 40 types
                soil_type_percentages = {f"type_{i + 1}_%": 0 for i in range(40)}
                # Update the dictionary with actual percentages
                for i, count in enumerate(soil_type_counts):
                    soil_type_percentages[f"type_{i + 1}_%"] = count
                # Append the information to the list
                soil_info.append({
                    'sensor': sensor['name'],
                    'station': station['codi'],
                    'distance': distance,
                    **soil_type_percentages
                })
                pbar.update(1)
    # Create a DataFrame from the soil_info list
    final_df = pd.DataFrame(soil_info)
    if save:
        final_df.to_csv(final_data_dir_path + 'processed_icgc.csv', index=False)
    return final_df

def prepare_meteocat_data(save = True):
    print('Preparing Meteocat data...')
    meteocat_1000 = pd.read_csv(_var_code_paths['1000'])
    meteocat_1300 = pd.read_csv(_var_code_paths['1300'])
    meteocat_1600 = pd.read_csv(_var_code_paths['1600'])
    processed_meteocat = pd.concat([meteocat_1000, meteocat_1300, meteocat_1600])
    processed_meteocat['data'] = pd.to_datetime(processed_meteocat['data'], format='%Y-%m-%d')
    processed_meteocat.sort_values(by='data', inplace=True)
    if save:
        processed_meteocat.to_csv(final_data_dir_path + 'processed_meteocat.csv', index=False)

prepare_aca_data()
prepare_icgc_data()
prepare_meteocat_data()










































