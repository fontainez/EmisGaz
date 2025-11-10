"""
EmisGaz Data Preparation Pipeline
Comprehensive data cleaning and preparation for emissions and climate analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import warnings

warnings.filterwarnings("ignore")


class EmisGazDataPreparer:
    """
    Comprehensive data preparation class for EmisGaz project
    Handles cleaning, transformation, and integration of emissions and climate data
    """

    def __init__(self, file_path: str = "dataset.xlsx", results_dir: str = "results"):
        self.file_path = file_path
        self.results_dir = results_dir
        self.excel_file = None
        self.cleaned_data = {}

    def load_dataset(self) -> None:
        """Load the Excel file and get sheet information"""
        print("📂 Loading dataset...")
        self.excel_file = pd.ExcelFile(self.file_path)
        print(f"✅ Dataset loaded successfully")
        print(f"📊 Sheets available: {self.excel_file.sheet_names}")

    def prepare_emissions_by_sector(self) -> pd.DataFrame:
        """
        Prepare the main emissions by sector data (2010-2020)
        Returns cleaned DataFrame with sectors as index and years as columns
        """
        print("\n🌍 Preparing emissions by sector data...")

        # Load the raw data
        df = pd.read_excel(self.file_path, sheet_name="emission_secteurs_ans")

        # Set the first column as index (sectors)
        df_clean = df.set_index("Année")

        # Convert French decimal format to Python floats
        df_clean = df_clean.replace(",", ".", regex=True)

        # Convert all year columns to numeric
        year_columns = [
            col
            for col in df_clean.columns
            if isinstance(col, (int, float)) and 2010 <= col <= 2020
        ]
        for col in year_columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

        # Clean up index name
        df_clean.index.name = "Sector"

        print(f"✅ Emissions data prepared: {df_clean.shape}")
        print(f"   Sectors: {list(df_clean.index)}")
        print(f"   Years: {list(df_clean.columns)}")

        return df_clean

    def prepare_greenhouse_gas_synthesis(
        self, period: str = "2010_2014"
    ) -> pd.DataFrame:
        """
        Prepare greenhouse gas synthesis data with detailed gas breakdown
        """
        print(f"\n📊 Preparing GHG synthesis data ({period})...")

        sheet_name = f"synthèse des GES {period}"
        df = pd.read_excel(self.file_path, sheet_name=sheet_name)

        # Clean the data
        df_clean = df.copy()

        # Handle missing values in Module column
        df_clean["Module"] = df_clean["Module"].fillna(method="ffill")

        # Convert French decimal format
        df_clean = df_clean.replace(",", ".", regex=True)

        # Identify year columns and percentage columns
        year_cols = []
        pct_cols = []

        for col in df_clean.columns:
            if isinstance(col, (int, float)) and 2010 <= col <= 2020:
                year_cols.append(col)
                df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
            elif "%" in str(col):
                pct_cols.append(col)
                df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

        print(f"✅ GHG synthesis prepared: {df_clean.shape}")
        print(f"   Year columns: {year_cols}")
        print(f"   Percentage columns: {pct_cols}")

        return df_clean

    def prepare_climate_data(self, data_type: str = "temperature") -> pd.DataFrame:
        """
        Prepare climate data (temperature or precipitation)
        Returns DataFrame with years as index and months as columns
        """
        print(f"\n🌡️ Preparing {data_type} data...")

        if data_type == "temperature":
            sheet_name = "Température 2010_2020"
            parameter_col = "PARAMETER"
        else:
            sheet_name = "Précipitation 2010-2020"
            parameter_col = "PARAMETER"

        # Load raw data
        df = pd.read_excel(self.file_path, sheet_name=sheet_name)

        # Remove parameter rows and set YEAR as index
        df_clean = df[df[parameter_col].isin(["T2M", "PRECTOTCORR"])].copy()
        df_clean = df_clean.set_index("YEAR")
        df_clean = df_clean.drop(columns=[parameter_col])

        # Ensure all values are numeric
        month_columns = [
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

        for col in month_columns:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

        print(f"✅ {data_type.capitalize()} data prepared: {df_clean.shape}")
        print(f"   Years: {df_clean.index.min()} - {df_clean.index.max()}")
        print(
            f"   Monthly columns: {[col for col in month_columns if col in df_clean.columns]}"
        )

        return df_clean

    def create_emissions_timeseries(self, emissions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert emissions data from wide to long format for time series analysis
        """
        print("\n📈 Creating emissions time series...")

        # Reset index to make sectors a column
        df_long = emissions_df.reset_index()

        # Melt the dataframe to create year-based time series
        df_timeseries = pd.melt(
            df_long,
            id_vars=["Sector"],
            value_vars=[
                col for col in emissions_df.columns if isinstance(col, (int, float))
            ],
            var_name="Year",
            value_name="Emissions",
        )

        # Pivot to get sectors as columns
        df_final = df_timeseries.pivot(
            index="Year", columns="Sector", values="Emissions"
        )
        df_final = df_final.sort_index()

        print(f"✅ Time series created: {df_final.shape}")
        print(f"   Years: {list(df_final.index)}")
        print(f"   Sectors: {list(df_final.columns)}")

        return df_final

    def create_annual_climate_summary(
        self, temp_df: pd.DataFrame, precip_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create annual climate summary combining temperature and precipitation
        """
        print("\n🌤️ Creating annual climate summary...")

        # Extract annual values
        climate_summary = pd.DataFrame()

        if "ANN" in temp_df.columns:
            climate_summary["Temperature_Annual"] = temp_df["ANN"]

        if "ANN" in precip_df.columns:
            climate_summary["Precipitation_Annual"] = precip_df["ANN"]

        # Add year as column for merging
        climate_summary["Year"] = climate_summary.index

        print(f"✅ Climate summary created: {climate_summary.shape}")

        return climate_summary

    def merge_emissions_climate_data(
        self, emissions_ts: pd.DataFrame, climate_summary: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge emissions and climate data for correlation analysis
        """
        print("\n🔗 Merging emissions and climate data...")

        # Prepare emissions data for merging
        emissions_for_merge = emissions_ts.reset_index()
        emissions_for_merge["Year"] = emissions_for_merge["Year"].astype(int)

        # Merge datasets
        merged_df = pd.merge(
            emissions_for_merge, climate_summary, on="Year", how="inner"
        )
        merged_df = merged_df.set_index("Year")

        print(f"✅ Data merged successfully: {merged_df.shape}")
        print(f"   Common years: {list(merged_df.index)}")
        print(f"   Variables: {list(merged_df.columns)}")

        return merged_df

    def calculate_emissions_totals(self, emissions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate total emissions and sector contributions
        """
        print("\n🧮 Calculating emissions totals and contributions...")

        # Calculate totals
        totals = pd.DataFrame()
        totals["Total_Emissions"] = emissions_df.sum(axis=1)

        # Calculate sector contributions (percentage)
        for sector in emissions_df.columns:
            if sector != "Total sans FAT":  # Avoid double counting
                totals[f"{sector}_Pct"] = (
                    emissions_df[sector] / totals["Total_Emissions"]
                ) * 100

        print(f"✅ Totals calculated: {totals.shape}")

        return totals

    def run_complete_preparation(self) -> Dict[str, pd.DataFrame]:
        """
        Run the complete data preparation pipeline
        Returns dictionary with all prepared datasets
        """
        print("🚀 Starting complete EmisGaz data preparation pipeline...")
        print("=" * 60)

        # Load dataset
        self.load_dataset()

        # Prepare emissions data
        emissions_by_sector = self.prepare_emissions_by_sector()
        emissions_timeseries = self.create_emissions_timeseries(emissions_by_sector)

        # Prepare GHG synthesis data
        ghg_2010_2014 = self.prepare_greenhouse_gas_synthesis("2010_2014")
        ghg_2015_2020 = self.prepare_greenhouse_gas_synthesis("2015_2020")

        # Prepare climate data
        temperature_data = self.prepare_climate_data("temperature")
        precipitation_data = self.prepare_climate_data("precipitation")

        # Create summaries and merge
        climate_summary = self.create_annual_climate_summary(
            temperature_data, precipitation_data
        )
        emissions_totals = self.calculate_emissions_totals(emissions_timeseries)

        # Merge for analysis
        merged_data = self.merge_emissions_climate_data(
            emissions_timeseries, climate_summary
        )

        # Store all prepared data
        self.cleaned_data = {
            "emissions_by_sector": emissions_by_sector,
            "emissions_timeseries": emissions_timeseries,
            "emissions_totals": emissions_totals,
            "ghg_2010_2014": ghg_2010_2014,
            "ghg_2015_2020": ghg_2015_2020,
            "temperature": temperature_data,
            "precipitation": precipitation_data,
            "climate_summary": climate_summary,
            "merged_analysis": merged_data,
        }

        print("\n" + "=" * 60)
        print("✅ DATA PREPARATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\n📊 Prepared datasets:")
        for name, df in self.cleaned_data.items():
            print(f"   📁 {name}: {df.shape}")

        return self.cleaned_data

    def save_prepared_data(self, output_path: str = "prepared_data.xlsx") -> None:
        """
        Save all prepared data to an Excel file
        """
        if not self.cleaned_data:
            print("❌ No data to save. Run preparation first.")
            return

        # Ensure results directory exists
        import os

        os.makedirs(self.results_dir, exist_ok=True)

        # Save to results directory
        full_output_path = os.path.join(self.results_dir, output_path)

        print(f"\n💾 Saving prepared data to {full_output_path}...")

        with pd.ExcelWriter(full_output_path, engine="openpyxl") as writer:
            for sheet_name, df in self.cleaned_data.items():
                df.to_excel(writer, sheet_name=sheet_name)

        print("✅ Data saved successfully!")


def main():
    """
    Main execution function for the data preparation pipeline
    """
    # Initialize the preparer
    preparer = EmisGazDataPreparer(results_dir="results")

    # Run complete preparation
    prepared_data = preparer.run_complete_preparation()

    # Save the prepared data
    preparer.save_prepared_data()

    # Display sample of key datasets
    print("\n" + "=" * 60)
    print("📋 SAMPLE DATA PREVIEW")
    print("=" * 60)

    print("\n📈 Emissions Time Series (first 3 years):")
    print(prepared_data["emissions_timeseries"].head(3))

    print("\n🌡️ Climate Summary (first 3 years):")
    print(prepared_data["climate_summary"].head(3))

    print("\n🔗 Merged Analysis Data (first 3 years):")
    print(prepared_data["merged_analysis"].head(3))


if __name__ == "__main__":
    main()
