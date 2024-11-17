# This document will be the main script for data extraction (automatically).
from datetime import datetime
import aca
import meteocat as m
import utils

log_trigger = 'Manual'

# Update data from ACA
aca.join_daily_data_with_all_data(aca.transform_daily_data(aca.get_daily_data()), overwrite=True, log_trigger=log_trigger)
print('ACA data updated successfully')

# Update data from Meteocat
meteocat_vars = ["1300", "1000", "1600"]
today_vars = {
    '1300': '35',
    '1000': '32',
    '1600': '38'
}
current_month = datetime.now().month - 1 if datetime.now().day == 1 else datetime.now().month
current_year = datetime.now().year
for var in meteocat_vars:
    m.log_meteocat_data(var, current_year, current_month, trigger=log_trigger)
    today_values = m.add_today_information(today_vars[var])
    daily_values = m.transform_daily_data(m.get_daily_data(var, current_year, current_month))
    merged_values = m.join_daily_and_today_data(daily_values, today_values)
    m.join_meteocat_data(m.data_getter(var) ,merged_values, overwrite=True)
    print(f'Meteocat data for variable {var} updated successfully')

# Update today data files
utils.update_today_data_file()
print('Today data files updated successfully')