import datetime

import pandas as pd
import requests

import utils, sanity_check

"""
This file is used to extract daily data from ACA API.
As real time data is only available for 3 months, the oldest data is downloaded from: https://analisi.transparenciacatalunya.cat/Medi-Ambient/Quantitat-d-aigua-als-embassaments-de-les-Conques-/gn9e-3qhr/about_data
"""

# Base url for metadata
_CATALOG_BASE_URL = "http://aca-web.gencat.cat/sdim2/apirest/catalog?componentType="
# Base url for data
_EMBASSAMENT_DATA_BASE_URL = ["http://aca-web.gencat.cat/sdim2/apirest/data/EMBASSAMENT-EST/", "?limit=-1&from=", "T00:00:00&to=", "T23:59:59"]

# As the names of the embassaments are not the same in the metadata and in the data, we need to map them
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
    """
    This function returns all the sensor codes from the metadata file
    :param sensor_data_file: path to the metadata file
    :return: list of sensor codes
    """
    df = pd.read_csv(utils.get_root_dir() + '/' + sensor_data_file)
    return df['sensor'].tolist()

def _get_metadata():
    """
    This function returns the metadata of the sensors
    :return: DataFrame with the metadata
    """
    return pd.read_csv(utils.get_root_dir() + '/data/processed/aca/sensor_metadata.csv')

def _get_all_data():
    """
    This function returns all the data from the aca_daily_all.csv file
    :return: DataFrame with all the data
    """
    return pd.read_csv(utils.get_root_dir() + '/data/processed/aca/aca_daily_all.csv')

def get_daily_data(date_from=None, date_to=None, sensor_code = 'all'):
    """
    This function returns the daily data from the ACA API
    :param date_from: Date from which we want to get the data (format: dd/mm/yyyy)
    :param date_to: Date to which we want to get the data (format: dd/mm/yyyy)
    :param sensor_code: Sensor code from which we want to get the data. If 'all', it will get the data from all the sensors
    :return: DataFrame with the daily data
    """
    # If dates are None, it takes date_from as yesterday and date_to as today
    if date_from is None:
        date_from = utils.get_date('yesterday')
    if date_to is None:
        date_to = utils.get_date('today')

    data_cols = ['value', 'timestamp', 'location', 'sensor']
    df_all_data = pd.DataFrame(columns=data_cols) # All retrieved data will be stored here

    if sensor_code == 'all':
        sensor_codes = get_all_sensor_codes()
    else:
        sensor_codes = [sensor_code] if type(sensor_code) == str else sensor_code

    # Retrieve data from all the sensors
    for sensor in sensor_codes:
        url = _EMBASSAMENT_DATA_BASE_URL[0] + sensor + _EMBASSAMENT_DATA_BASE_URL[1] + date_from + _EMBASSAMENT_DATA_BASE_URL[2] + date_to + _EMBASSAMENT_DATA_BASE_URL[3]
        response = requests.get(url)
        df = pd.json_normalize(response.json(), record_path=['observations'])
        df['sensor'] = sensor
        # Concatenate the data to get a unique DataFrame
        df_all_data = pd.concat([df_all_data, df])
    return df_all_data

def transform_daily_data(df, only_most_recent = True):
    """
    This function transforms the daily data DataFrame to match the format of the aca_daily_all.csv file
    :param df: DataFrame with the daily data
    :param only_most_recent: If True, it will only keep the most recent data from each day
    :return: DataFrame with the transformed data
    """
    # Reformat daily_data_df date and value types
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d/%m/%YT%H:%M:%S')
    df['value'] = df['value'].astype(float)
    df['date'] = df['timestamp'].dt.date
    df['time'] = df['timestamp'].dt.time
    # Add sensor type and drop useless columns
    df['sensor_type'] = df['sensor'].map(_get_metadata().set_index('sensor')['sensor_type'])
    df.drop(columns=['timestamp', 'location'], inplace=True)
    df.sort_values(by=['date', 'time'], ascending=False, inplace=True)
    # Keep most recent if only_most_recent is True
    if only_most_recent:
        df = df.groupby(['date', 'sensor', 'sensor_type']).first().reset_index()
    # Add name to the sensors
    df['name'] = df['sensor'].map(_get_metadata().set_index('sensor')['name'])

    # Pivot table to get the data in the right format
    df = df.pivot_table(index=['date', 'name'], columns='sensor_type', values='value').reset_index()
    metadata_ = _get_metadata()
    # Add sensor codes to the DataFrame
    df['sensor_current_volume'] = df['name'].map(metadata_[metadata_['sensor_type'] == 'current_volume'].set_index('name')['sensor'])
    df['sensor_total_volume'] = df['name'].map(metadata_[metadata_['sensor_type'] == 'total_volume'].set_index('name')['sensor'])
    df['sensor_percentage'] = df['name'].map(metadata_[metadata_['sensor_type'] == 'percentage'].set_index('name')['sensor'])
    return df

def join_daily_data_with_all_data(daily_data_df, all_data_df=_get_all_data(), overwrite=True, log_trigger='Manual'):
    """
    This function joins the daily data with the all data DataFrame
    :param daily_data_df: daily data DataFrame
    :param all_data_df: all data DataFrame
    :param overwrite: If True, it will overwrite the aca_daily_all.csv file
    :param log_trigger: Trigger of the log. It can be 'Manual' or 'Automatic'
    :return: None if overwrite is True, DataFrame with the updated data if overwrite is False
    """
    # Only keep the data that is not in the daily data DataFrame
    all_data_df['date'] = pd.to_datetime(all_data_df['date'])
    daily_data_df['date'] = pd.to_datetime(daily_data_df['date'])
    all_data_df = all_data_df[~all_data_df['date'].isin(daily_data_df['date'])].copy()
    # Concatenate the DataFrames
    data_updated = pd.concat([all_data_df, daily_data_df], ignore_index=True)
    data_updated.drop_duplicates(inplace=True)
    data_updated.sort_values(by=['date', 'name'], ascending=False, inplace=True)
    data_updated.reset_index(drop=True, inplace=True)
    # Save the data or return it
    if overwrite:
        utils.save_df_to_csv(data_updated, "aca_daily_all", "data/processed/aca/")
    else:
        return data_updated
    # Log the update
    log_msg = "Data updated with daily data from " + datetime.datetime.strftime(daily_data_df['date'].min(), '%d/%m/%Y') + " to " + datetime.datetime.strftime(daily_data_df['date'].max(), '%d/%m/%Y') + " -  Embassaments: " + str(';'.join(daily_data_df['name'].unique()))
    sanity_check.log_auto_aca_data_update(trigger=log_trigger, msg=log_msg)

def get_catalog(catalog_type="embassament"):
    """
    This function returns the catalog of the sensors. Used to update the metadata
    :param catalog_type: Type of the catalog. Default is 'embassament'
    :return: DataFrame with the catalog
    """
    url = _CATALOG_BASE_URL + catalog_type
    response = requests.get(url)
    return pd.json_normalize(response.json(), record_path=['providers', 'sensors'])

def update_aca_metadata():
    """
    This function updates the metadata file of the sensors
    :return: None
    """
    utils.save_df_to_csv(get_catalog(), "embassaments_metadata", "DataExtraction/metadata/aca_metadata/")

def transform_metadata(aca_metadata_src = "DataExtraction/metadata/aca_metadata/embassaments_metadata.csv", save_path = "data/processed/aca/", save_name = "sensor_metadata", update=True):
    """
    This function transforms the metadata file to consume it in the data extraction process
    :param aca_metadata_src: Path to the raw metadata file
    :param save_path: Path to save the transformed metadata file
    :param save_name: Name of the transformed metadata file
    :param update: If True, it will update the raw metadata file before running
    :return: None
    """
    if update:
        update_aca_metadata()
    aca_metadata_src = utils.get_root_dir() + "/" + aca_metadata_src
    df = pd.read_csv(aca_metadata_src, dtype={'location': 'str'})
    # Mapping of the sensor types
    mapping_sensor_types = {
        'Volum embassat': 'current_volume',
        'Nivell absolut': 'total_volume',
        'Percentatge volum embassat': 'percentage'
    }
    df['sensor_type'] = df['description'].map(mapping_sensor_types)
    df = df[pd.notna(df['sensor_type'])][['componentDesc', 'location', 'sensor', 'sensor_type']]
    # Split location into latitude and longitude
    df['latitude'] = df['location'].apply(lambda x: x.split(" ")[0])
    df['longitude'] = df['location'].apply(lambda x: x.split(" ")[1])
    df.drop(columns=['location'], inplace=True)
    df.drop_duplicates(inplace=True)
    df.columns = ['name', 'sensor', 'sensor_type', 'latitude', 'longitude']
    # Add the name
    df['name'] = df['name'].apply(lambda x: x.strip()).map(_NAME_MATCHING_DICT)
    utils.save_df_to_csv(df, save_name, save_path)

def transform_historical_data(aca_historical_data_src = "data/raw/aca/manual_download/historical_data.csv", save_path = "data/processed/aca/", save_name = "aca_daily_all", update_metadata=True):
    """
    This function transforms the raw historical data to final format
    :param aca_historical_data_src: Path to the raw historical data
    :param save_path: Path to save the transformed data
    :param save_name: Name of the transformed data
    :param update_metadata: If True, it will update the metadata before running (RECOMMENDED)
    :return: None
    """
    # Access raw data file (updated manually)
    aca_historical_data_src = utils.get_root_dir() + "/" + aca_historical_data_src
    df = pd.read_csv(aca_historical_data_src, dtype={'Dia': 'str'})
    df['Dia'] = df['Dia'].apply(lambda x: utils.parse_date(x, "%d/%m/%Y"))
    df.sort_values(by=['Dia'], ascending=True, inplace=True)
    # Rename columns
    df.columns = ['date', 'name', 'total_volume', 'percentage', 'current_volume']
    # Update metadata
    if update_metadata:
        transform_metadata()
    df_metadata = _get_metadata()
    df['sensor_current_volume'] = df.merge(df_metadata[df_metadata['sensor_type'] == 'current_volume'], on='name', how='left')['sensor'].values
    df['sensor_total_volume'] = df.merge(df_metadata[df_metadata['sensor_type'] == 'total_volume'], on='name', how='left')['sensor'].values
    df['sensor_percentage'] = df.merge(df_metadata[df_metadata['sensor_type'] == 'percentage'], on='name', how='left')['sensor'].values
    utils.save_df_to_csv(df, save_name, save_path)


# update_aca_metadata() # --> To update embassaments_metadata.csv
# transform_metadata() # --> To transform embassaments_metadata.csv
# transform_historical_data() # --> To transform historical_data.csv
# join_daily_data_with_all_data(transform_daily_data(get_daily_data())) # --> To update aca_daily_all.csv