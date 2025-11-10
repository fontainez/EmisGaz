"""
In this file, we will clean the data and prepare it for analysis.
We will clean the data by:
1. Load each sheet into a pandas dataframe
2. Clean column name and data types
3. Convert French-style numbers (,) into pyhton floats
4. Create clean DataFrame for: 
    - Emissions (2010-2020)
    - Precipitation (1981-2020)
    - Temperature (1981-2020)
"""

# Let import necessary Library
import pandas as pd

# Define file path
file_path = 'dataset.xlsx'

# Function to load and clean a sheet
def load_and_clean_sheet(sheet_name, index_col=0):
    df = pd.read_excel(file_path, sheet_name=sheet_name, index_col=index_col)
    
    # Replace commas with dots for decimal conversion
    df = df.replace(',', '.', regex=True)
    
    # Convert all columns to numeric (coerce errors to NaN)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Safely rename index if it has a name
    if df.index.name is not None:
        if 'Année' in df.index.name or df.index.name == "Unnamed: 0":
            df.index.name = 'Category'
    
    return df

# Load and clean key sheets
print("Loading and cleaning emissions data...")
df_emissions = load_and_clean_sheet('emission_secteurs_ans')




print("\nLoading and cleaning temperature data...")
df_temp_raw = pd.read_excel(file_path, sheet_name='Température 2010_2020', index_col=None)

# Clean: remove first row if it's 'PARAMETER' or similar
if 'PARAMETER' in df_temp_raw.iloc[0].values:
    df_temp_raw = df_temp_raw.drop(0).reset_index(drop=True)

# Set YEAR as index
df_temp_raw = df_temp_raw.set_index('YEAR') # Thi transpose the dataframe(year become rows and months become columns)

# Remove PARAMETER column if it exists
if 'PARAMETER' in df_temp_raw.columns:
    df_temp_raw = df_temp_raw.drop(columns=['PARAMETER'])

# DO NOT TRANSPOSE — we want years as rows, months as columns
df_temp = df_temp_raw

# Ensure all values are float
for col in df_temp.columns:
    df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce')




print("\nLoading and cleaning precipitation data...")
df_precip_raw = pd.read_excel(file_path, sheet_name='Précipitation 2010-2020', index_col=None)

# Clean: remove first row if it's 'PARAMETER' or similar
if 'PARAMETER' in df_precip_raw.iloc[0].values:
    df_precip_raw = df_precip_raw.drop(0).reset_index(drop=True)

# Set YEAR as index
df_precip_raw = df_precip_raw.set_index('YEAR')

# Remove PARAMETER column if it exists
if 'PARAMETER' in df_precip_raw.columns:
    df_precip_raw = df_precip_raw.drop(columns=['PARAMETER'])

# DO NOT TRANSPOSE — we want years as rows, months as columns
df_precip = df_precip_raw

# Ensure all values are float
for col in df_precip.columns:
    df_precip[col] = pd.to_numeric(df_precip[col], errors='coerce')


# Now let display cleaned data info
print("\n--- CLEANED TARGET DATA INFO ---")

"""print("\nEmissions (2010–2020):")
print(df_emissions.info())
print(df_emissions.head())
print(f"\nShape: {df_emissions.shape}")
"""
print("\nTemperature (1981–2020):")
#print(df_temp.info())
print(df_temp.head())
print(f"\nShape: {df_temp.shape}")

"""print("\nPrecipitation (1981–2020):")
print(df_precip.info())
#print(df_precip.head())
print(f"\nShape: {df_precip.shape}")"""