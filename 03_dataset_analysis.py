"""
EmisGaz Dataset Analysis Script
Comprehensive analysis of the EmisGaz dataset to understand structure and content
"""

import pandas as pd
import numpy as np
from datetime import datetime


def analyze_dataset():
    """Main function to perform comprehensive dataset analysis"""

    print("=== EMISGAZ DATASET COMPREHENSIVE ANALYSIS ===\n")

    # Load the dataset
    file_path = "dataset.xlsx"

    try:
        excel_file = pd.ExcelFile(file_path)
        print(f"✅ Dataset loaded successfully: {file_path}")
        print(f"📊 Number of sheets: {len(excel_file.sheet_names)}\n")

        # Display all sheet names
        print("📋 SHEET OVERVIEW:")
        for i, sheet in enumerate(excel_file.sheet_names):
            print(f"  {i}: {sheet}")

        print("\n" + "=" * 80 + "\n")

        # Analyze each sheet in detail
        for sheet_name in excel_file.sheet_names:
            analyze_sheet(sheet_name, file_path)

    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return


def analyze_sheet(sheet_name, file_path):
    """Analyze a specific sheet in detail"""

    print(f"🔍 ANALYZING SHEET: '{sheet_name}'")
    print("-" * 50)

    try:
        # Load the sheet
        df = pd.read_excel(file_path, sheet_name=sheet_name)

        # Basic information
        print(f"📐 Shape: {df.shape} (rows: {df.shape[0]}, columns: {df.shape[1]})")
        print(f"📝 Data types:")
        print(df.dtypes.value_counts())

        # Display first few rows
        print(f"\n📄 First 5 rows:")
        print(df.head())

        # Check for missing values
        missing_values = df.isnull().sum()
        total_missing = missing_values.sum()
        print(f"\n❓ Missing values: {total_missing} total")
        if total_missing > 0:
            print("Columns with missing values:")
            for col, count in missing_values[missing_values > 0].items():
                print(f"  - {col}: {count} missing ({count / len(df) * 100:.1f}%)")

        # Check for duplicate rows
        duplicates = df.duplicated().sum()
        print(f"\n🔄 Duplicate rows: {duplicates}")

        # Column analysis
        print(f"\n📊 Column analysis:")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        text_cols = df.select_dtypes(include=["object"]).columns

        print(f"  - Numeric columns: {len(numeric_cols)}")
        print(f"  - Text columns: {len(text_cols)}")

        if len(numeric_cols) > 0:
            print(f"\n📈 Numeric columns summary:")
            print(df[numeric_cols].describe())

        # Special analysis based on sheet name
        if "emission" in sheet_name.lower() or "ges" in sheet_name.lower():
            analyze_emissions_sheet(df, sheet_name)
        elif "température" in sheet_name.lower() or "temperature" in sheet_name.lower():
            analyze_temperature_sheet(df, sheet_name)
        elif (
            "précipitation" in sheet_name.lower()
            or "precipitation" in sheet_name.lower()
        ):
            analyze_precipitation_sheet(df, sheet_name)
        elif "description" in sheet_name.lower():
            analyze_description_sheet(df, sheet_name)

        print("\n" + "=" * 80 + "\n")

    except Exception as e:
        print(f"❌ Error analyzing sheet '{sheet_name}': {e}")
        print("\n" + "=" * 80 + "\n")


def analyze_emissions_sheet(df, sheet_name):
    """Specialized analysis for emissions sheets"""
    print(f"🌍 EMISSIONS SHEET ANALYSIS: {sheet_name}")

    # Check for year columns
    year_cols = [
        col
        for col in df.columns
        if isinstance(col, (int, float)) and 2010 <= col <= 2020
    ]
    if year_cols:
        print(f"  - Year columns found: {sorted(year_cols)}")

    # Check for gas types
    gas_keywords = ["CO2", "CH4", "N2O", "Energie", "Agriculture", "PIUP"]
    gas_cols = []
    for col in df.columns:
        if any(keyword in str(col) for keyword in gas_keywords):
            gas_cols.append(col)

    if gas_cols:
        print(f"  - Gas/Sector related columns: {gas_cols}")

    # Look for percentage columns
    pct_cols = [col for col in df.columns if "%" in str(col)]
    if pct_cols:
        print(f"  - Percentage columns: {pct_cols}")


def analyze_temperature_sheet(df, sheet_name):
    """Specialized analysis for temperature sheets"""
    print(f"🌡️ TEMPERATURE SHEET ANALYSIS: {sheet_name}")

    # Check for month columns
    month_cols = [
        col
        for col in df.columns
        if isinstance(col, str)
        and col.upper()
        in [
            "JAN",
            "FEB",
            "MAR",
            "APR",
            "MAY",
            "JUN",
            "JUL",
            "AUG",
            "SEP",
            "OCT",
            "NOV",
            "DEC",
            "ANN",
        ]
    ]
    if month_cols:
        print(f"  - Month columns found: {month_cols}")

    # Check for year index
    if "YEAR" in df.columns:
        print(f"  - YEAR column found")
    elif df.index.name == "YEAR":
        print(f"  - YEAR index found")


def analyze_precipitation_sheet(df, sheet_name):
    """Specialized analysis for precipitation sheets"""
    print(f"🌧️ PRECIPITATION SHEET ANALYSIS: {sheet_name}")

    # Similar to temperature analysis
    month_cols = [
        col
        for col in df.columns
        if isinstance(col, str)
        and col.upper()
        in [
            "JAN",
            "FEB",
            "MAR",
            "APR",
            "MAY",
            "JUN",
            "JUL",
            "AUG",
            "SEP",
            "OCT",
            "NOV",
            "DEC",
            "ANN",
        ]
    ]
    if month_cols:
        print(f"  - Month columns found: {month_cols}")

    if "YEAR" in df.columns:
        print(f"  - YEAR column found")
    elif df.index.name == "YEAR":
        print(f"  - YEAR index found")


def analyze_description_sheet(df, sheet_name):
    """Specialized analysis for description/model sheets"""
    print(f"📖 DESCRIPTION SHEET ANALYSIS: {sheet_name}")

    # Display all content for description sheets
    print("  - Full content:")
    for idx, row in df.iterrows():
        print(f"    Row {idx}: {row.to_dict()}")


def generate_data_dictionary():
    """Generate a comprehensive data dictionary"""
    print("\n📚 DATA DICTIONARY GENERATION")
    print("=" * 50)

    file_path = "dataset.xlsx"
    excel_file = pd.ExcelFile(file_path)

    data_dict = {}

    for sheet_name in excel_file.sheet_names:
        print(f"\n📋 Sheet: {sheet_name}")
        df = pd.read_excel(file_path, sheet_name=sheet_name)

        sheet_info = {
            "description": get_sheet_description(sheet_name),
            "columns": {},
            "sample_data": df.head(3).to_dict(),
        }

        for col in df.columns:
            col_info = {
                "data_type": str(df[col].dtype),
                "non_null_count": df[col].count(),
                "null_count": df[col].isnull().sum(),
                "unique_values": df[col].nunique()
                if df[col].dtype == "object"
                else "N/A",
                "sample_values": df[col].dropna().head(3).tolist()
                if len(df[col].dropna()) > 0
                else [],
            }

            if df[col].dtype in [np.int64, np.float64]:
                col_info.update(
                    {
                        "min": df[col].min(),
                        "max": df[col].max(),
                        "mean": df[col].mean(),
                        "std": df[col].std(),
                    }
                )

            sheet_info["columns"][str(col)] = col_info

        data_dict[sheet_name] = sheet_info

        # Print summary for this sheet
        print(f"  Description: {sheet_info['description']}")
        print(f"  Columns: {len(df.columns)}")
        print(f"  Rows: {len(df)}")

    return data_dict


def get_sheet_description(sheet_name):
    """Get description for each sheet based on name"""
    descriptions = {
        "Description du model": "Model description and methodology",
        "emission_secteurs_ans": "Annual emissions by sector (2010-2020)",
        "synthèse des GES 2010_2014": "Greenhouse gas synthesis 2010-2014",
        "synthèse des GES 2015_2020": "Greenhouse gas synthesis 2015-2020",
        "Température 2010_2020": "Temperature data 2010-2020",
        "Précipitation 2010-2020": "Precipitation data 2010-2020",
    }

    return descriptions.get(sheet_name, "No description available")


def main():
    """Main execution function"""
    print("Starting EmisGaz Dataset Analysis...")
    print("=" * 60)

    # Perform comprehensive analysis
    analyze_dataset()

    # Generate data dictionary
    data_dict = generate_data_dictionary()

    print("\n✅ Analysis completed successfully!")
    print(f"📅 Analysis timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
