from datetime import datetime
import aca
import meteocat as m
import utils

"""
This script updates the data from ACA and Meteocat and saves it to the corresponding files.
Used to update automatically the data every day.
Can be executed manually to update the data (change the log_trigger to 'Manual') or automatically by the scheduler.
"""

_log_trigger = 'Manual'

# Update data from ACA
print('Updating ACA data...')
aca.join_daily_data_with_all_data(aca.transform_daily_data(aca.get_daily_data()), overwrite=True, log_trigger=_log_trigger)
print('ACA data updated successfully')

# Update data from Meteocat
meteocat_vars = ["1300", "1000", "1600"]
today_vars = {
    '1300': '35',
    '1000': '32',
    '1600': '38'
}
current_month = datetime.now().month - 1 if datetime.now().day == 1 else datetime.now().month
current_month = "1" if current_month == 0 else str(current_month)
current_year = str(datetime.now().year)
for var in meteocat_vars:
    print(f'Updating Meteocat data for variable {var}...')
    m.log_meteocat_data(var, current_year, current_month, trigger=_log_trigger)
    today_values = m.add_today_information(today_vars[var])
    daily_values = m.transform_daily_data(m.get_daily_data(var, current_year, current_month))
    merged_values = m.join_daily_and_today_data(daily_values, today_values)
    m.join_meteocat_data(m.data_getter(var) ,merged_values, overwrite=True)
    print(f'Meteocat data for variable {var} updated successfully')

# Update today data files
utils.update_today_data_file()