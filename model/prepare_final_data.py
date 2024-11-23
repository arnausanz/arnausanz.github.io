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
    if save:
        processed_aca.to_csv(final_data_dir_path + 'processed_aca.csv', index=False)
    return processed_aca

def prepare_meteocat_data(save = True):
    print('Preparing Meteocat data...')
    meteocat_1000 = pd.read_csv(_var_code_paths['1000'])
    meteocat_1300 = pd.read_csv(_var_code_paths['1300'])
    meteocat_1600 = pd.read_csv(_var_code_paths['1600'])
    processed_meteocat = pd.concat([meteocat_1000, meteocat_1300, meteocat_1600])
    processed_meteocat['data'] = pd.to_datetime(processed_meteocat['data'], format='%Y-%m-%d')
    processed_meteocat.sort_values(by='data', inplace=True)
    # Drop all Z8 station data as it does not appear in the metadata
    processed_meteocat = processed_meteocat[processed_meteocat['codiEstacio'] != 'Z8']
    # Pivot the table to have the variables as columns
    processed_meteocat = processed_meteocat.pivot(index=['data', 'codiEstacio'], columns='codiVariable', values='valor').reset_index()
    # Create all day-station combination in case there are missing days for some stations
    date_range = pd.date_range(start=processed_meteocat['data'].min(), end=processed_meteocat['data'].max())
    # Drop stations with more than 50% missing days
    station_to_keep = processed_meteocat.groupby('codiEstacio').size() / (len(date_range))
    station_to_keep = station_to_keep[station_to_keep > 0.5].index
    processed_meteocat = processed_meteocat[processed_meteocat['codiEstacio'].isin(station_to_keep)]
    # Create a final dataframe with all the combinations of keeping stations and days
    stations = processed_meteocat['codiEstacio'].unique()
    all_combinations = pd.MultiIndex.from_product([date_range, stations], names=['data', 'codiEstacio'])
    all_days_stations = pd.DataFrame(index=all_combinations).reset_index()
    # Merge the all_days_stations with the processed_meteocat
    processed_meteocat = pd.merge(all_days_stations, processed_meteocat, on=['data', 'codiEstacio'], how='left')
    # Set codiEstacio as pd.Categorical and save the codes-names mapping into a new csv file
    processed_meteocat['codiEstacio'] = pd.Categorical(processed_meteocat['codiEstacio'])
    processed_meteocat['station_code'] = processed_meteocat['codiEstacio'].cat.codes
    processed_meteocat[['station_code', 'codiEstacio']].drop_duplicates().to_csv(final_data_dir_path + 'station_codes.csv', index=False)
    # Drop codiEstacio
    processed_meteocat.drop(columns='codiEstacio', inplace=True)
    # Fill missing values with the previous day value
    processed_meteocat.ffill(inplace=True)
    # Fill any remaining missing values with the next day value
    processed_meteocat.bfill(inplace=True)
    # Reset index
    processed_meteocat.reset_index(drop=True, inplace=True)
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
    # Use station and sensor codes from saved csv files
    station_codes = pd.read_csv(final_data_dir_path + 'station_codes.csv')
    sensor_codes = pd.read_csv(final_data_dir_path + 'sensor_codes.csv')
    final_df = pd.merge(final_df, station_codes, left_on='station', right_on='codiEstacio', how='left')
    final_df = pd.merge(final_df, sensor_codes, left_on='sensor', right_on='name', how='left')
    final_df.drop(columns=['station', 'sensor', 'codiEstacio', 'name'], inplace=True)
    # Drop rows with missing station or sensor codes (means that we dropped them earlier)
    final_df.dropna(subset=['station_code', 'sensor_code'], inplace=True)
    # Keep sensor and station codes as ints
    final_df['station_code'] = final_df['station_code'].astype(int)
    final_df['sensor_code'] = final_df['sensor_code'].astype(int)
    # Set sensor and station codes as index to apply PCA
    final_df.set_index(['sensor_code', 'station_code'], inplace=True)
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

def update_data(save = True, update_aca = True, update_meteocat = True, update_icgc = False):
    prepare_aca_data(save) if update_aca else None
    prepare_meteocat_data(save) if update_meteocat else None
    prepare_icgc_data(save) if update_icgc else None
    print('Data updated successfully')
    # Merge data to get the final dataframe format for the model
    # Columns: date, sensor_code, station_code, 1000, 1300, 1600, pca_1, pca_2, pca_3, pca_4, pca_5, current_volume
    processed_aca = pd.read_csv(final_data_dir_path + 'processed_aca.csv')
    processed_meteocat = pd.read_csv(final_data_dir_path + 'processed_meteocat.csv')
    processed_icgc = pd.read_csv(final_data_dir_path + 'processed_icgc.csv')
    final_data = pd.merge(processed_aca, processed_meteocat, left_on='date', right_on='data', how='inner')
    final_data.drop(columns='data', inplace=True)
    final_data = pd.merge(final_data, processed_icgc, on=['sensor_code', 'station_code'], how='inner')
    # Reorder columns
    final_data = final_data[['date', 'sensor_code', 'station_code', '1000', '1300', '1600', 'pca_1', 'pca_2', 'pca_3', 'pca_4', 'current_volume']]
    if save:
        final_data.to_csv(final_data_dir_path + 'final_data.csv', index=False)

def get_data(update = False, save = True, update_aca = True, update_meteocat = True, update_icgc = False):
    if update:
        update_data(save, update_aca, update_meteocat, update_icgc)
    data = pd.read_csv(final_data_dir_path + 'final_data.csv')
    data['sensor_code'] = data['sensor_code'].astype('int64')
    data['station_code'] = data['station_code'].astype('int64')
    data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')
    return data


































