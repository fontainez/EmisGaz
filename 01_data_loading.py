# load the dataset and print out what sheets exist.

import pandas as pd

# load the dataset
file_path = "dataset.xlsx"
excel_file = pd.ExcelFile(file_path)

# Print all the sheets in the excel file
for i, sheet in enumerate(excel_file.sheet_names):
    print(f"sheet {i}: {sheet}")

# Try reading the first few rows of the 'emission_secteurs_ans' sheet
try:
    df_emissions = pd.read_excel(file_path, sheet_name="emission_secteurs_ans")
    print("\nFirst few rows of 'emission_secteurs_ans':")
    print(df_emissions.head(10))
except Exception as e:
    print(f"Error reading emission_secteurs_ans: {e}")
