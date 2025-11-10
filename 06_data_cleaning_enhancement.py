"""
EmisGaz Data Cleaning Enhancement
Enhanced cleaning to fix identified data quality issues from validation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings("ignore")


class EmisGazDataCleaner:
    """
    Enhanced data cleaning class for EmisGaz project
    Fixes identified issues from data validation
    """

    def __init__(self, file_path: str = "dataset.xlsx", results_dir: str = "results"):
        self.file_path = file_path
        self.results_dir = results_dir
        self.cleaned_data = {}

    def fix_emissions_negative_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fix negative values in emissions data by taking absolute values
        where negative values don't make sense
        """
        print("🔧 Fixing negative emissions values...")

        df_clean = df.copy()

        # Identify numeric columns (years)
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns

        # For emissions data, negative values don't make sense
        # Take absolute values for all numeric columns
        for col in numeric_cols:
            if (df_clean[col] < 0).any():
                negative_count = (df_clean[col] < 0).sum()
                print(
                    f"  - {col}: {negative_count} negative values -> converting to positive"
                )
                df_clean[col] = df_clean[col].abs()

        return df_clean

    def fix_ghg_synthesis_missing_values(
        self, df: pd.DataFrame, sheet_name: str
    ) -> pd.DataFrame:
        """
        Fix missing values in GHG synthesis sheets using appropriate methods
        """
        print(f"🔧 Fixing missing values in {sheet_name}...")

        df_clean = df.copy()

        # Fill missing Module values using forward fill
        if "Module" in df_clean.columns:
            missing_before = df_clean["Module"].isnull().sum()
            df_clean["Module"] = df_clean["Module"].fillna(method="ffill")
            missing_after = df_clean["Module"].isnull().sum()
            print(
                f"  - Module: {missing_before} missing -> {missing_after} missing after forward fill"
            )

        # For numeric columns, use interpolation for missing values
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if df_clean[col].isnull().sum() > 0:
                missing_before = df_clean[col].isnull().sum()
                # Use linear interpolation for time series data
                df_clean[col] = df_clean[col].interpolate(
                    method="linear", limit_direction="both"
                )
                missing_after = df_clean[col].isnull().sum()
                print(
                    f"  - {col}: {missing_before} missing -> {missing_after} missing after interpolation"
                )

        return df_clean

    def remove_duplicate_rows(
        self, df: pd.DataFrame, dataset_name: str
    ) -> pd.DataFrame:
        """
        Remove duplicate rows from datasets
        """
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            print(f"🔧 Removing {duplicates} duplicate rows from {dataset_name}")
            df_clean = df.drop_duplicates()
            return df_clean
        return df

    def enhance_emissions_by_sector(self) -> pd.DataFrame:
        """
        Enhanced cleaning for emissions by sector data
        """
        print("\n🌍 Enhancing emissions by sector data...")

        # Load raw data
        df = pd.read_excel(self.file_path, sheet_name="emission_secteurs_ans")

        # Set sectors as index
        df_clean = df.set_index("Année")
        df_clean.index.name = "Sector"

        # Convert French decimal format
        df_clean = df_clean.replace(",", ".", regex=True)

        # Convert to numeric
        year_cols = [col for col in df_clean.columns if isinstance(col, (int, float))]
        for col in year_cols:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

        # Fix negative values
        df_clean = self.fix_emissions_negative_values(df_clean)

        print(f"✅ Enhanced emissions data: {df_clean.shape}")
        return df_clean

    def enhance_ghg_synthesis(self, period: str) -> pd.DataFrame:
        """
        Enhanced cleaning for GHG synthesis data
        """
        print(f"\n📊 Enhancing GHG synthesis data ({period})...")

        sheet_name = f"synthèse des GES {period}"
        df = pd.read_excel(self.file_path, sheet_name=sheet_name)

        # Remove duplicates
        df_clean = self.remove_duplicate_rows(df, sheet_name)

        # Fill missing Module values
        df_clean["Module"] = df_clean["Module"].fillna(method="ffill")

        # Convert French decimal format
        df_clean = df_clean.replace(",", ".", regex=True)

        # Convert to numeric
        year_cols = [
            col
            for col in df_clean.columns
            if isinstance(col, (int, float)) and 2010 <= col <= 2020
        ]
        pct_cols = [col for col in df_clean.columns if "%" in str(col)]

        for col in year_cols + pct_cols:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

        # Fix missing values
        df_clean = self.fix_ghg_synthesis_missing_values(df_clean, sheet_name)

        print(f"✅ Enhanced GHG synthesis: {df_clean.shape}")
        return df_clean

    def create_enhanced_emissions_timeseries(
        self, emissions_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create enhanced emissions time series with proper data types
        """
        print("\n📈 Creating enhanced emissions time series...")

        # Reset index for melting
        df_long = emissions_df.reset_index()

        # Melt to long format
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

        # Ensure all values are positive (emissions can't be negative)
        for col in df_final.columns:
            if (df_final[col] < 0).any():
                df_final[col] = df_final[col].abs()

        print(f"✅ Enhanced time series: {df_final.shape}")
        return df_final

    def calculate_enhanced_totals(self, emissions_ts: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate enhanced emissions totals with proper handling
        """
        print("\n🧮 Calculating enhanced emissions totals...")

        totals = pd.DataFrame()

        # Calculate total emissions (sum of all sectors except totals)
        sectors_to_sum = [col for col in emissions_ts.columns if "Total" not in col]
        totals["Total_Emissions"] = emissions_ts[sectors_to_sum].sum(axis=1)

        # Calculate sector contributions (percentage)
        for sector in sectors_to_sum:
            totals[f"{sector}_Pct"] = (
                emissions_ts[sector] / totals["Total_Emissions"]
            ) * 100

        # Handle division by zero
        totals = totals.replace([np.inf, -np.inf], np.nan)

        print(f"✅ Enhanced totals calculated: {totals.shape}")
        return totals

    def enhance_climate_data(self, data_type: str) -> pd.DataFrame:
        """
        Enhanced cleaning for climate data (already clean, but adding validation)
        """
        print(f"\n🌡️ Enhancing {data_type} data...")

        if data_type == "temperature":
            sheet_name = "Température 2010_2020"
        else:
            sheet_name = "Précipitation 2010-2020"

        # Load and clean as before
        df = pd.read_excel(self.file_path, sheet_name=sheet_name)

        # Remove parameter rows and set YEAR as index
        parameter_col = "PARAMETER"
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

        print(f"✅ Enhanced {data_type} data: {df_clean.shape}")
        return df_clean

    def run_enhanced_cleaning(self) -> Dict[str, pd.DataFrame]:
        """
        Run the complete enhanced data cleaning pipeline
        """
        print("🚀 Starting enhanced EmisGaz data cleaning pipeline...")
        print("=" * 60)

        # Enhanced emissions data
        emissions_by_sector = self.enhance_emissions_by_sector()
        emissions_timeseries = self.create_enhanced_emissions_timeseries(
            emissions_by_sector
        )
        emissions_totals = self.calculate_enhanced_totals(emissions_timeseries)

        # Enhanced GHG synthesis
        ghg_2010_2014 = self.enhance_ghg_synthesis("2010_2014")
        ghg_2015_2020 = self.enhance_ghg_synthesis("2015_2020")

        # Enhanced climate data
        temperature_data = self.enhance_climate_data("temperature")
        precipitation_data = self.enhance_climate_data("precipitation")

        # Create climate summary
        climate_summary = pd.DataFrame()
        if "ANN" in temperature_data.columns:
            climate_summary["Temperature_Annual"] = temperature_data["ANN"]
        if "ANN" in precipitation_data.columns:
            climate_summary["Precipitation_Annual"] = precipitation_data["ANN"]
        climate_summary["Year"] = climate_summary.index

        # Merge for analysis
        emissions_for_merge = emissions_timeseries.reset_index()
        emissions_for_merge["Year"] = emissions_for_merge["Year"].astype(int)
        merged_data = pd.merge(
            emissions_for_merge, climate_summary, on="Year", how="inner"
        )
        merged_data = merged_data.set_index("Year")

        # Store all enhanced data
        self.cleaned_data = {
            "emissions_by_sector_enhanced": emissions_by_sector,
            "emissions_timeseries_enhanced": emissions_timeseries,
            "emissions_totals_enhanced": emissions_totals,
            "ghg_2010_2014_enhanced": ghg_2010_2014,
            "ghg_2015_2020_enhanced": ghg_2015_2020,
            "temperature_enhanced": temperature_data,
            "precipitation_enhanced": precipitation_data,
            "climate_summary_enhanced": climate_summary,
            "merged_analysis_enhanced": merged_data,
        }

        print("\n" + "=" * 60)
        print("✅ ENHANCED DATA CLEANING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\n📊 Enhanced datasets:")
        for name, df in self.cleaned_data.items():
            print(f"   📁 {name}: {df.shape}")

        return self.cleaned_data

    def save_enhanced_data(self, output_path: str = "enhanced_data.xlsx") -> None:
        """
        Save all enhanced data to an Excel file
        """
        if not self.cleaned_data:
            print("❌ No enhanced data to save. Run cleaning first.")
            return

        # Ensure results directory exists
        import os

        os.makedirs(self.results_dir, exist_ok=True)

        # Save to results directory
        full_output_path = os.path.join(self.results_dir, output_path)

        print(f"\n💾 Saving enhanced data to {full_output_path}...")

        with pd.ExcelWriter(full_output_path, engine="openpyxl") as writer:
            for sheet_name, df in self.cleaned_data.items():
                df.to_excel(writer, sheet_name=sheet_name)

        print("✅ Enhanced data saved successfully!")


def main():
    """
    Main execution function for enhanced data cleaning
    """
    # Initialize the cleaner
    cleaner = EmisGazDataCleaner(results_dir="results")

    # Run enhanced cleaning
    enhanced_data = cleaner.run_enhanced_cleaning()

    # Save the enhanced data
    cleaner.save_enhanced_data()

    # Display sample of enhanced datasets
    print("\n" + "=" * 60)
    print("📋 ENHANCED DATA PREVIEW")
    print("=" * 60)

    print("\n📈 Enhanced Emissions Time Series (first 3 years):")
    print(enhanced_data["emissions_timeseries_enhanced"].head(3))

    print("\n🧮 Enhanced Emissions Totals (first 3 years):")
    print(enhanced_data["emissions_totals_enhanced"].head(3))

    print("\n🔗 Enhanced Merged Analysis Data (first 3 years):")
    print(enhanced_data["merged_analysis_enhanced"].head(3))


if __name__ == "__main__":
    main()
