import numpy as np
from geopy.distance import geodesic
import geopandas as gpd
from scipy.spatial import cKDTree
from shapely.geometry.linestring import LineString
from tqdm import tqdm
from DataExtraction import utils
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


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
    # Set name as pd.Categorical and save the codes-names mapping into a new csv file
    processed_aca['name'] = pd.Categorical(processed_aca['name'])
    processed_aca['sensor_code'] = processed_aca['name'].cat.codes
    processed_aca[['sensor_code', 'name']].drop_duplicates().to_csv(final_data_dir_path + 'sensor_codes.csv', index=False)
    # Drop name
    processed_aca.drop(columns='name', inplace=True)
    # Use ffills to fill missing values if there's any
    processed_aca.ffill(inplace=True)
    # Keep a single row per day and one colum for each sensor
    processed_aca = processed_aca.pivot(index='date', columns='sensor_code', values='current_volume').reset_index()
    if save:
        processed_aca.to_csv(final_data_dir_path + 'processed_aca.csv', index=False)
    return processed_aca

def prepare_meteocat_data(save = True):
    print('Preparing Meteocat data...')
    meteocat_1000 = pd.read_csv(_var_code_paths['1000'])
    # Drop all Z8 station data as it does not appear in the metadata
    meteocat_1000 = meteocat_1000[meteocat_1000['codiEstacio'] != 'Z8']
    meteocat_1000 = meteocat_1000.pivot(index='data', columns='codiEstacio', values='valor').reset_index()
    meteocat_1000.columns = ['date'] + [f'1000_{col}' for col in meteocat_1000.columns[1:]]
    meteocat_1300 = pd.read_csv(_var_code_paths['1300'])
    # Drop all Z8 station data as it does not appear in the metadata
    meteocat_1300 = meteocat_1300[meteocat_1300['codiEstacio'] != 'Z8']
    meteocat_1300 = meteocat_1300.pivot(index='data', columns='codiEstacio', values='valor').reset_index()
    meteocat_1300.columns = ['date'] + [f'1300_{col}' for col in meteocat_1300.columns[1:]]
    meteocat_1600 = pd.read_csv(_var_code_paths['1600'])
    # Drop all Z8 station data as it does not appear in the metadata
    meteocat_1600 = meteocat_1600[meteocat_1600['codiEstacio'] != 'Z8']
    meteocat_1600 = meteocat_1600.pivot(index='data', columns='codiEstacio', values='valor').reset_index()
    meteocat_1600.columns = ['date'] + [f'1600_{col}' for col in meteocat_1600.columns[1:]]
    # Merge the three dataframes
    processed_meteocat = pd.merge(meteocat_1000, meteocat_1300, on='date', how='outer')
    processed_meteocat = pd.merge(processed_meteocat, meteocat_1600, on='date', how='outer')
    processed_meteocat['date'] = pd.to_datetime(processed_meteocat['date'], format='%Y-%m-%d')
    processed_meteocat.sort_values(by='date', inplace=True)
    # Create missing days if there are any
    date_range = pd.date_range(start=processed_meteocat['date'].min(), end=processed_meteocat['date'].max())
    # Add missing days
    processed_meteocat = processed_meteocat.set_index('date').reindex(date_range).reset_index()
    # Rename index to date
    processed_meteocat.rename(columns={'index': 'date'}, inplace=True)
    # Drop columns with more than 50% of Nan values
    processed_meteocat.dropna(thresh=0.75*len(processed_meteocat), axis=1, inplace=True)
    # Fill nan values with the previous day value
    processed_meteocat.ffill(inplace=True)
    # Fill any remaining missing values with the next day value
    processed_meteocat.bfill(inplace=True)
    print(processed_meteocat.shape)
    if save:
        processed_meteocat.to_csv(final_data_dir_path + 'processed_meteocat.csv', index=False)

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
                    num_points = 1 # If a sensor is too close to a station, we still want to get some information
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
    # Drop colums with all 0 values
    final_df = final_df.loc[:, (final_df != 0).any(axis=0)]
    final_df.set_index(['sensor', 'station'], inplace=True)
    """
    ###############################################
    To detect the best number of components in PCA
    ###############################################
    pca = PCA()
    pca.fit(final_df)
    cumulative_explained_variance = pca.explained_variance_ratio_.cumsum()
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o', linestyle='--')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Explained Variance vs. Number of Components')
    plt.grid(True)
    plt.show()
    """
    # Apply PCA with 4 components (elbow technique)
    pca = PCA(n_components=4)
    final_df = pd.DataFrame(pca.fit_transform(final_df), index=final_df.index)
    # Rename pca columns
    final_df.columns = [f'pca_{i + 1}' for i in range(4)]
    # Reset index
    final_df.reset_index(inplace=True)
    if save:
        final_df.to_csv(final_data_dir_path + 'processed_icgc.csv', index=False)
    return final_df

def update_data(save = True, with_icgc = False):
    prepare_aca_data(save)
    prepare_meteocat_data(save)
    prepare_icgc_data(save) if with_icgc else None
    print('Data updated successfully')
    # Merge data to get the final dataframe format for the model
    # Columns: date, sensor_code, station_code, 1000, 1300, 1600, pca_1, pca_2, pca_3, pca_4, pca_5, current_volume
    processed_aca = pd.read_csv(final_data_dir_path + 'processed_aca.csv')
    processed_meteocat = pd.read_csv(final_data_dir_path + 'processed_meteocat.csv')
    processed_icgc = pd.read_csv(final_data_dir_path + 'processed_icgc.csv') if with_icgc else None
    final_data = pd.merge(processed_aca, processed_meteocat, on='date', how='inner')
    if with_icgc:
        final_data = pd.merge(final_data, processed_icgc, on=['sensor_code', 'station_code'], how='inner')
    if save:
        final_data.to_csv(final_data_dir_path + 'final_data.csv', index=False)

def get_data(update = False, save = True, with_icgc = False):
    if update:
        update_data(save, with_icgc)
    data = pd.read_csv(final_data_dir_path + 'final_data.csv')
    data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')
    return data