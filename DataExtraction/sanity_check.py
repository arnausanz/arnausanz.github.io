import datetime

import pandas as pd

import utils
"""
This file is aimed to check the sanity of all the process (e.g. the data extraction, the data join, etc.)

It will be run periodically (every day at 00:00) to check that everything is working as expected.

Also, here we have the logs implementation of automatic data updates
"""

def check_aca_data_sanity():
    # TODO --> Implement a function that ckecks the data file. If data file is not correct, make a secure copy of the file and keep working on the original
    # Also throws an error if the data is not correct (to make github send error email)
    pass

def check_meteocat_data_sanity():
    # TODO --> Implement a function that ckecks the data file. If data file is not correct, make a secure copy of the file and keep working on the original
    # Also throws an error if the data is not correct (to make github send error email)
    pass

def log_auto_aca_data_update(trigger, msg, log_file = 'logs/aca_data_update.txt'):
    with open(utils.get_root_dir() + '/' + log_file, 'a') as f:
        f.write(f"{datetime.datetime.strftime(datetime.datetime.now(), '%d-%m-%Y %H:%M:%S')} - Updated by: {trigger} - {msg}\n")
        f.close()

def save_backup(file, backup_path = 'data/backup/'):
    utils.save_df_to_csv(pd.read_csv(file), 'backup_' + file.split('/')[-1].split('.')[0] + '_' + datetime.datetime.strftime(datetime.datetime.today(), '%d_%m_%Y'), backup_path)

# save_backup(utils.get_root_dir() + '/' + 'data/processed/aca/aca_daily_all.csv') # --> Save aca_daily_all.csv backup