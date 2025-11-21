# Data and script folder stems
import os 

## Enter path to directory containing uk data here:
data_folder_uk = 'UPDATE THIS'

## Enter path to directory containing US data here:
data_folder_us = 'UPDATE THIS'

# Database and asset locations (uk)
db_pth_uk = os.path.join(data_folder_uk, 'gambling_data.db')

asset_pq_uk = os.path.join(data_folder_uk, 'uk_asset_long.parquet')
pdf_pth_uk = os.path.join(data_folder_uk, 'uk_asset.parquet')

pop_asset_pq_uk = os.path.join(data_folder_uk, 'uk_population_asset_long.parquet') # Full population
pop_pdf_pth_uk = os.path.join(data_folder_uk, 'uk_population_asset.parquet')

asset_pq_gross_uk = os.path.join(data_folder_uk, 'uk_asset_long_gross.parquet')
pdf_pth_gross_uk = os.path.join(data_folder_uk, 'uk_asset_gross.parquet')

asset_deb_and_cred_pq_uk = os.path.join(data_folder_uk, 'uk_adf_deb_and_cred.parquet')
asset_deb_cred_net_pq_uk = os.path.join(data_folder_uk, 'uk_deb_cred_net.parquet')

asset_pq_uk_coffee = os.path.join(data_folder_uk, 'adf_coffee.parquet')
pdf_pth_uk_coffee = os.path.join(data_folder_uk, 'pdf_coffee.parquet')

# Database and asset location (US)
db_pth_us = os.path.join(data_folder_us, 'data.db')

asset_pq_us = os.path.join(data_folder_us, 'us_asset_long.parquet')
pdf_pth_us = os.path.join(data_folder_us, 'us_asset.parquet')

pop_asset_pq_us = os.path.join(data_folder_us, 'us_population_asset_long.parquet') # Full population
pop_pdf_pth_us = os.path.join(data_folder_us, 'us_population_asset.parquet')

asset_pq_gross_us = os.path.join(data_folder_us, 'us_asset_long_gross.parquet')
pdf_pth_gross_us = os.path.join(data_folder_us, 'us_asset_gross.parquet')

asset_deb_and_cred_pq_us = os.path.join(data_folder_us, 'us_adf_deb_and_cred.parquet')
asset_deb_cred_net_pq_us = os.path.join(data_folder_us, 'us_deb_cred_net.parquet')

# Outputs
output_folder = f'outputs'

# Lookups
lkp_xlsx = f'lookups/COICOP_maps.xlsx'
gva_tables_xlsx = f'lookups/gva_tables.xlsx'

# SQL
sql_folder = f'src/sql'
spend_sql = f'{sql_folder}/get_spend.sql'
savings_sql = f'{sql_folder}/get_savings.sql'
credit_card_sql = f'{sql_folder}/get_credit_card_interest.sql'
coffee_sql = f'{sql_folder}/get_coffee.sql'