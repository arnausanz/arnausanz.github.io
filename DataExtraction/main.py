# This document will be the main script for data extraction (automatically).
import aca

# Update data from ACA
aca.join_daily_data_with_all_data(aca.transform_daily_data(aca.get_daily_data()), overwrite=True, log_trigger='Automatic')
