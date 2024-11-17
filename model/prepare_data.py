import pandas as pd
import numpy as np
from geopy.distance import geodesic
import DataExtraction.utils as utils
import geopandas as gpd
from shapely.geometry import LineString, Point
from tqdm import tqdm
from scipy.spatial import cKDTree
from sklearn.preprocessing import MinMaxScaler

_var_code_paths = {
    'aca': utils.get_root_dir() + '/data/processed/aca/aca_daily_all.csv',
    '1300': utils.get_root_dir() + '/data/processed/meteocat/meteocat_1300_daily_all.csv',
    '1000': utils.get_root_dir() + '/data/processed/meteocat/meteocat_1000_daily_all.csv',
    '1600': utils.get_root_dir() + '/data/processed/meteocat/meteocat_1600_daily_all.csv',
    'icgc': utils.get_root_dir() + '/data/processed/icgc/cobertes_sol.csv'
}

def meteocat_data(var_code):
    data = pd.read_csv(_var_code_paths[var_code])
    data['data'] = pd.to_datetime(data['data'], format='%Y-%m-%d')
    data.sort_values(by='data', inplace=True)
    data.drop(columns=['codiVariable'], inplace=True)
    # Keep only actual working stations from meteocat/stations_metadata.csv
    stations_metadata = pd.read_csv(utils.get_root_dir() + '/data/processed/meteocat/stations_metadata.csv')
    data = data[data['codiEstacio'].isin(stations_metadata['codi'])]
    return data

def aca_data_getter():
    data = pd.read_csv(_var_code_paths['aca'])
    data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')
    data.sort_values(by='date', inplace=True)
    # For now, we'll only use the current_volume data (as our y)
    data = data[['date', 'name', 'current_volume']].copy()
    return data

def icgc_data():
    data = pd.read_csv(_var_code_paths['icgc'])
    return data

def read_metadata():
    _meteocat_stations_metadata = pd.read_csv(utils.get_root_dir() + '/data/processed/meteocat/stations_metadata.csv')
    _aca_sensors_metadata = pd.read_csv(utils.get_root_dir() + '/data/processed/aca/sensor_metadata.csv')
    _meteocat_stations_metadata['coordenades.latitud'] = _meteocat_stations_metadata['coordenades.latitud'].astype(np.float64)
    _meteocat_stations_metadata['coordenades.longitud'] = _meteocat_stations_metadata['coordenades.longitud'].astype(np.float64)
    _aca_sensors_metadata['latitude'] = _aca_sensors_metadata['latitude'].astype(np.float64)
    _aca_sensors_metadata['longitude'] = _aca_sensors_metadata['longitude'].astype(np.float64)
    _aca_sensors_metadata = _aca_sensors_metadata[['name', 'latitude', 'longitude']].copy()
    _aca_sensors_metadata.drop_duplicates(inplace=True)
    return _aca_sensors_metadata, _meteocat_stations_metadata

# TODO --> Això no pot estar per aquí al mig
aca_sensors_metadata, meteocat_stations_metadata = read_metadata()
soil_data = icgc_data()

def calc_distances():
    """
    Calculate the distances between all the stations for each reservoir
    :return: pd.DataFrame
    """
    distances = []
    for _, aca_sensor in aca_sensors_metadata.iterrows():
        for _, meteocat_station in meteocat_stations_metadata.iterrows():
            distance = geodesic((aca_sensor['latitude'], aca_sensor['longitude']), (meteocat_station['coordenades.latitud'], meteocat_station['coordenades.longitud'])).km
            distances.append((aca_sensor['name'], meteocat_station['codi'], distance))

    return pd.DataFrame(distances, columns=['aca_sensor', 'meteocat_station', 'distance'])

def get_geodataframes():
    _geo_stations = gpd.GeoDataFrame(meteocat_stations_metadata, geometry=gpd.points_from_xy(meteocat_stations_metadata['coordenades.longitud'], meteocat_stations_metadata['coordenades.latitud']), crs='EPSG:4326')
    _geo_sensors = gpd.GeoDataFrame(aca_sensors_metadata, geometry=gpd.points_from_xy(aca_sensors_metadata['longitude'], aca_sensors_metadata['latitude']), crs='EPSG:4326')
    _geo_soil = gpd.GeoDataFrame(soil_data, geometry=gpd.points_from_xy(soil_data['Longitude'], soil_data['Latitude']), crs='EPSG:4326')
    return _geo_stations, _geo_sensors, _geo_soil

# I have the 3 geodataframes
geo_stations, geo_sensors, geo_soil = get_geodataframes()


def get_soil_information_between_points():
    """
    Function to get the soil information of the points between the sensors and the stations. It searches for the closest point in the soil data
    :return: pd.DataFrame
    """
    # Re-project geometries to a projected CRS
    geo_stations_proj = geo_stations.to_crs(epsg=3857)
    geo_sensors_proj = geo_sensors.to_crs(epsg=3857)
    geo_soil_proj = geo_soil.to_crs(epsg=3857)

    # Create a spatial index for the soil data
    soil_tree = cKDTree(np.vstack([geo_soil_proj.geometry.x, geo_soil_proj.geometry.y]).T)

    soil_information = []
    with tqdm(total=geo_sensors_proj.shape[0], desc="Processing sensors") as pbar_sensors:
        for _, sensor in geo_sensors_proj.iterrows():
            for _, station in geo_stations_proj.iterrows():
                # Get the line between the sensor and the station
                line = LineString([sensor['geometry'], station['geometry']])
                # Calculate the distance between the sensor and the station
                distance = geodesic((sensor['latitude'], sensor['longitude']),
                                    (station['coordenades.latitud'], station['coordenades.longitud'])).km
                # Interpolate points along the line
                num_points = int(distance * 10)  # Number of points to interpolate, adjust as needed
                if num_points == 0:
                    continue  # Skip if there are no points to interpolate
                interpolated_points = [line.interpolate(i / num_points, normalized=True) for i in range(num_points + 1)]
                # Find the nearest soil type for each interpolated point
                soil_types = []
                for point in interpolated_points:
                    point_coords = np.array([point.x, point.y])
                    _, idx = soil_tree.query(point_coords)
                    soil_types.append(geo_soil_proj.iloc[idx]['Class'])
                # Calculate the percentage of each soil type
                soil_type_counts = pd.Series(soil_types).value_counts(normalize=True) * 100
                # Initialize the soil type percentages with 0 for all 40 types
                soil_type_percentages = {f"type_{i + 1}_%": 0 for i in range(40)}
                # Update the dictionary with actual percentages
                for i, count in enumerate(soil_type_counts):
                    soil_type_percentages[f"type_{i + 1}_%"] = count
                # Append the information to the list
                soil_information.append({
                    'sensor': sensor['name'],
                    'station': station['codi'],
                    'distance': distance,
                    **soil_type_percentages
                })
            pbar_sensors.update(1)

    return pd.DataFrame(soil_information)


def transform_meteocat_dataframes(dataframe, var):
    """
    This function transforms the meteocat dataframes to get the following structure:
    Index will be the date, columns will be the stations, and the values will be the variables
    :return: DataFrame transformed
    """
    # Pivot the dataframe
    df = dataframe.pivot(index='data', columns='codiEstacio', values='valor')
    # Rename the columns
    df.columns = [f"{col}_{var}" for col in df.columns]
    # Ensure the index is set to the date
    df.index.name = 'date'
    return df

def merge_all_meteocat_data():
    meteocat_data_1000 = meteocat_data('1000')
    meteocat_data_1300 = meteocat_data('1300')
    meteocat_data_1600 = meteocat_data('1600')
    meteocat_data_1000_transformed = transform_meteocat_dataframes(meteocat_data_1000, '1000')
    meteocat_data_1300_transformed = transform_meteocat_dataframes(meteocat_data_1300, '1300')
    meteocat_data_1600_transformed = transform_meteocat_dataframes(meteocat_data_1600, '1600')

    # Concatenate the dataframes along the columns
    meteocat_merged = pd.concat(
        [meteocat_data_1000_transformed, meteocat_data_1300_transformed, meteocat_data_1600_transformed], axis=1)
    meteocat_merged.reset_index(inplace=True)

    return meteocat_merged

def get_meteocat_merged_data():
    m_merged = pd.read_csv(utils.get_root_dir() + '/model/data_prepared/meteocat_merged.csv')
    m_merged['date'] = pd.to_datetime(m_merged['date'], format='%Y-%m-%d')
    m_merged.set_index('date', inplace=True)
    return m_merged

def merge_all_meteocat_data_aca():
    # Get aca data
    aca_data = aca_data_getter()
    aca_data_transformed = aca_data.pivot(index='date', columns='name', values='current_volume')
    aca_data_transformed.index.name = 'date'
    # Concat with merged meteocat data
    meteocat_merged = get_meteocat_merged_data()

    # Ensure both indices are datetime and normalized to dates
    meteocat_merged.index = pd.to_datetime(meteocat_merged.index)
    aca_data_transformed.index = pd.to_datetime(aca_data_transformed.index)

    # Concatenate the dataframes along the columns
    merged_data = pd.concat([meteocat_merged, aca_data_transformed], axis=1)
    return merged_data


def get_merged_final_data_aca_meteocat():
    return pd.read_csv(utils.get_root_dir() + '/model/data_prepared/aca_meteocat_merged.csv')


def nan_treatment(_df):
    # Merge all data
    _df = merge_all_meteocat_data_aca()
    # Fill Nan values with 0 for all 1300 and 1600 variables. For 1000 variables, fill them using interpolate
    columns_1300 = [col for col in _df.columns if '1300' in col]
    columns_1600 = [col for col in _df.columns if '1600' in col]
    columns_1000 = [col for col in _df.columns if '1000' in col]

    _df[columns_1300] = _df[columns_1300].fillna(0)
    _df[columns_1600] = _df[columns_1600].fillna(0)
    _df[columns_1000] = _df[columns_1000].interpolate(method='time').ffill().bfill()

    # Fill nan values for aca data with same strategy as 1000 variables
    aca_columns = [col for col in _df.columns if 'Embassament' in col]
    _df[aca_columns] = _df[aca_columns].interpolate(method='time').ffill().bfill()

    return _df

def scale_data(_df):
    # Scale the data
    scaler = MinMaxScaler()
    _df_scaled = _df.copy()
    _df_scaled[_df_scaled.columns] = scaler.fit_transform(_df_scaled[_df_scaled.columns])
    return _df_scaled, scaler


def create_temporal_windows(_df, steps):
    """
    Generate temporal windows from a DataFrame for LSTM input.

    Parameters:
    - df (pd.DataFrame): The scaled DataFrame containing features and targets.
    - target_columns (list): List of column names representing the target variables.
    - steps (int): Number of time steps for the window.

    Returns:
    - X (np.ndarray): Input features with shape (samples, steps, features).
    - y (np.ndarray): Target values with shape (samples, targets).
    """
    # Ensure DataFrame is sorted by time (if necessary)
    _df = _df.sort_index()
    target_columns = [col for col in _df.columns if 'Embassament' in col]

    # Separate features and targets
    features = _df.drop(columns=target_columns).values
    targets = _df[target_columns].values

    X, y = [], []

    # Generate windows
    for i in range(len(_df) - steps):
        X.append(features[i:i + steps])  # Collect window of features
        y.append(targets[i + steps])  # Target corresponds to the end of the window

    return np.array(X), np.array(y)

def get_data_prepared(temporal_window, recalc_with_new_data = False, return_scaler = False):
    if recalc_with_new_data:
        # Update meteocat data and aca data and merge them
        utils.save_df_to_csv(merge_all_meteocat_data(), 'meteocat_merged', utils.get_root_dir() + '/model/data_prepared/')
        utils.save_df_to_csv(merge_all_meteocat_data_aca(), 'aca_meteocat_merged', utils.get_root_dir() + '/model/data_prepared/')
    df = get_merged_final_data_aca_meteocat()
    df = nan_treatment(df)
    df_scaled, _ = scale_data(df)
    X, y = create_temporal_windows(df_scaled, temporal_window)
    if return_scaler:
        return X, y, _
    else:
        return X, y


# TODO --> Netejar això

# Create the distances file
# utils.save_df_to_csv(calc_distances(), 'distances', utils.get_root_dir() + '/model/data_prepared/')

# Transform the distances file into a distance and soil types file
# utils.save_df_to_csv(get_soil_information_between_points(), 'soil_information', utils.get_root_dir() + '/model/data_prepared/')

# Merge all meteocat data
# utils.save_df_to_csv(merge_all_meteocat_data(), 'meteocat_merged', utils.get_root_dir() + '/model/data_prepared/')

# Get final big DataFrame of aca and meteocat data
# utils.save_df_to_csv(merge_all_meteocat_data_aca(), 'aca_meteocat_merged', utils.get_root_dir() + '/model/data_prepared/')



# df = nan_treatment(get_merged_final_data_aca_meteocat())
# print(df.head())
# df_scaled, _ = scale_data(df)
# print(df_scaled.head())

# X, y = create_temporal_windows(df_scaled, 14)
# print(X, y)

# X, y = get_data_prepared()
# print(X, y)