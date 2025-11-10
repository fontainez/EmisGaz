"""
EmisGaz Exploratory Data Analysis (EDA)
Comprehensive analysis of emissions and climate data relationships
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

# Set up plotting style
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


class EmisGazEDA:
    """
    Comprehensive Exploratory Data Analysis for EmisGaz project
    Analyzes emissions and climate data relationships
    """

    def __init__(self, data_path="enhanced_data.xlsx", results_dir="results"):
        self.data_path = data_path
        self.results_dir = results_dir
        self.datasets = {}
        self.analysis_results = {}

    def load_data(self):
        """Load all enhanced datasets"""
        print("📂 Loading enhanced data for EDA...")
        try:
            self.datasets = pd.read_excel(self.data_path, sheet_name=None)
            print(f"✅ Loaded {len(self.datasets)} datasets")

            # Ensure results directory exists
            import os

            os.makedirs(self.results_dir, exist_ok=True)
            print(f"📁 Results will be saved to: {self.results_dir}/")

            # Display dataset overview
            print("\n📊 Dataset Overview:")
            for name, df in self.datasets.items():
                print(f"   {name}: {df.shape}")

        except Exception as e:
            print(f"❌ Error loading data: {e}")

    def analyze_emissions_trends(self):
        """Analyze emissions trends over time"""
        print("\n📈 Analyzing Emissions Trends...")

        df = self.datasets["emissions_timeseries_enhanced"]

        # Basic statistics
        print("📊 Emissions Statistics (2010-2020):")
        stats_summary = df.describe()
        print(stats_summary)

        # Trend analysis
        print("\n📈 Emissions Growth (2010-2020):")
        growth_rates = {}
        for sector in df.columns:
            if sector not in ["Year", "Total avec FAT", "Total sans FAT"]:
                start_val = df[sector].iloc[0]
                end_val = df[sector].iloc[-1]
                growth_pct = ((end_val - start_val) / start_val) * 100
                growth_rates[sector] = growth_pct
                print(f"   {sector}: {growth_pct:+.1f}%")

        self.analysis_results["emissions_growth"] = growth_rates

        # Create trend visualization
        plt.figure(figsize=(12, 8))
        for sector in df.columns:
            if sector not in ["Year", "Total avec FAT", "Total sans FAT"]:
                plt.plot(df.index, df[sector], marker="o", linewidth=2, label=sector)

        plt.title(
            "Emissions Trends by Sector (2010-2020)", fontsize=16, fontweight="bold"
        )
        plt.xlabel("Year", fontsize=12)
        plt.ylabel("Emissions", fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            f"{self.results_dir}/emissions_trends.png", dpi=300, bbox_inches="tight"
        )
        plt.show()

    def analyze_sector_contributions(self):
        """Analyze sector contributions to total emissions"""
        print("\n🏭 Analyzing Sector Contributions...")

        df_totals = self.datasets["emissions_totals_enhanced"]

        # Average contributions over the period
        avg_contributions = df_totals.filter(like="_Pct").mean()

        print("📊 Average Sector Contributions (%):")
        for sector, contribution in avg_contributions.items():
            sector_name = sector.replace("_Pct", "")
            print(f"   {sector_name}: {contribution:.1f}%")

        # Create contribution visualization
        plt.figure(figsize=(10, 8))
        colors = plt.cm.Set3(np.linspace(0, 1, len(avg_contributions)))
        wedges, texts, autotexts = plt.pie(
            avg_contributions.values,
            labels=avg_contributions.index.str.replace("_Pct", ""),
            autopct="%1.1f%%",
            colors=colors,
            startangle=90,
        )

        plt.setp(autotexts, size=10, weight="bold")
        plt.setp(texts, size=10)
        plt.title(
            "Average Sector Contributions to Total Emissions (2010-2020)",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(
            f"{self.results_dir}/sector_contributions.png", dpi=300, bbox_inches="tight"
        )
        plt.show()

        self.analysis_results["sector_contributions"] = avg_contributions

    def analyze_climate_trends(self):
        """Analyze climate variable trends"""
        print("\n🌡️ Analyzing Climate Trends...")

        temp_df = self.datasets["temperature_enhanced"]
        precip_df = self.datasets["precipitation_enhanced"]

        # Focus on annual data
        temp_annual = temp_df["ANN"]
        precip_annual = precip_df["ANN"]

        # Temperature analysis
        print("🌡️ Temperature Analysis (1981-2020):")
        print(f"   Average: {temp_annual.mean():.2f}°C")
        print(
            f"   Trend: +{(temp_annual.iloc[-1] - temp_annual.iloc[0]):.2f}°C over 40 years"
        )
        print(f"   Standard Deviation: {temp_annual.std():.2f}°C")

        # Precipitation analysis
        print("\n🌧️ Precipitation Analysis (1981-2020):")
        print(f"   Average: {precip_annual.mean():.2f} units")
        print(
            f"   Trend: {(precip_annual.iloc[-1] - precip_annual.iloc[0]):.2f} units over 40 years"
        )
        print(f"   Standard Deviation: {precip_annual.std():.2f} units")

        # Create climate trend visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Temperature plot
        ax1.plot(temp_annual.index, temp_annual, color="red", linewidth=2, marker="o")
        ax1.set_title(
            "Annual Temperature Trend (1981-2020)", fontsize=14, fontweight="bold"
        )
        ax1.set_ylabel("Temperature (°C)", fontsize=12)
        ax1.grid(True, alpha=0.3)

        # Add trend line
        z = np.polyfit(temp_annual.index, temp_annual, 1)
        p = np.poly1d(z)
        ax1.plot(
            temp_annual.index,
            p(temp_annual.index),
            "r--",
            alpha=0.7,
            label=f"Trend: {z[0]:.4f}°C/year",
        )
        ax1.legend()

        # Precipitation plot
        ax2.plot(
            precip_annual.index, precip_annual, color="blue", linewidth=2, marker="o"
        )
        ax2.set_title(
            "Annual Precipitation Trend (1981-2020)", fontsize=14, fontweight="bold"
        )
        ax2.set_xlabel("Year", fontsize=12)
        ax2.set_ylabel("Precipitation", fontsize=12)
        ax2.grid(True, alpha=0.3)

        # Add trend line
        z_precip = np.polyfit(precip_annual.index, precip_annual, 1)
        p_precip = np.poly1d(z_precip)
        ax2.plot(
            precip_annual.index,
            p_precip(precip_annual.index),
            "b--",
            alpha=0.7,
            label=f"Trend: {z_precip[0]:.4f} units/year",
        )
        ax2.legend()

        plt.tight_layout()
        plt.savefig(
            f"{self.results_dir}/climate_trends.png", dpi=300, bbox_inches="tight"
        )
        plt.show()

        self.analysis_results["climate_trends"] = {
            "temp_trend": z[0],
            "precip_trend": z_precip[0],
        }

    def analyze_correlations(self):
        """Analyze correlations between emissions and climate variables"""
        print("\n🔗 Analyzing Emissions-Climate Correlations...")

        merged_df = self.datasets["merged_analysis_enhanced"]

        # Select relevant columns for correlation
        emissions_cols = [
            col
            for col in merged_df.columns
            if col not in ["Temperature_Annual", "Precipitation_Annual"]
        ]
        climate_cols = ["Temperature_Annual", "Precipitation_Annual"]

        # Calculate correlation matrix
        correlation_matrix = merged_df[emissions_cols + climate_cols].corr()

        # Extract emissions-climate correlations
        climate_correlations = correlation_matrix.loc[emissions_cols, climate_cols]

        print("📊 Emissions-Climate Correlations:")
        print(climate_correlations.round(3))

        # Create correlation heatmap
        plt.figure(figsize=(12, 8))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap="RdBu_r",
            center=0,
            square=True,
            fmt=".2f",
            cbar_kws={"shrink": 0.8},
        )

        plt.title(
            "Correlation Matrix: Emissions vs Climate Variables",
            fontsize=16,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(
            f"{self.results_dir}/correlation_matrix.png", dpi=300, bbox_inches="tight"
        )
        plt.show()

        # Create scatter plots for key relationships
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()

        key_sectors = ["Energie", "Agriculture", "Déchets", "PIUP"]

        for i, sector in enumerate(key_sectors):
            if i < 4:  # Ensure we don't exceed subplot count
                # Temperature vs Emissions
                axes[i].scatter(
                    merged_df[sector], merged_df["Temperature_Annual"], alpha=0.7, s=80
                )

                # Add trend line
                z = np.polyfit(merged_df[sector], merged_df["Temperature_Annual"], 1)
                p = np.poly1d(z)
                axes[i].plot(merged_df[sector], p(merged_df[sector]), "r--", alpha=0.8)

                corr_coef = merged_df[sector].corr(merged_df["Temperature_Annual"])
                axes[i].set_title(
                    f"{sector} vs Temperature\n(r = {corr_coef:.3f})",
                    fontsize=12,
                    fontweight="bold",
                )
                axes[i].set_xlabel(f"{sector} Emissions")
                axes[i].set_ylabel("Temperature (°C)")
                axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            f"{self.results_dir}/emissions_temperature_scatter.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

        self.analysis_results["correlations"] = climate_correlations

    def analyze_seasonal_patterns(self):
        """Analyze seasonal patterns in climate data"""
        print("\n🌤️ Analyzing Seasonal Patterns...")

        temp_df = self.datasets["temperature_enhanced"]
        precip_df = self.datasets["precipitation_enhanced"]

        # Monthly averages across all years
        months = [
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
        ]

        temp_monthly_avg = temp_df[months].mean()
        precip_monthly_avg = precip_df[months].mean()

        print("📊 Monthly Climate Averages:")
        monthly_data = pd.DataFrame(
            {"Temperature": temp_monthly_avg, "Precipitation": precip_monthly_avg}
        )
        print(monthly_data.round(2))

        # Create seasonal pattern visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Temperature seasonal pattern
        ax1.plot(
            range(1, 13),
            temp_monthly_avg.values,
            color="red",
            linewidth=3,
            marker="o",
            markersize=8,
        )
        ax1.set_title(
            "Average Monthly Temperature Pattern", fontsize=14, fontweight="bold"
        )
        ax1.set_ylabel("Temperature (°C)", fontsize=12)
        ax1.set_xticks(range(1, 13))
        ax1.set_xticklabels(months)
        ax1.grid(True, alpha=0.3)

        # Precipitation seasonal pattern
        ax2.bar(range(1, 13), precip_monthly_avg.values, color="blue", alpha=0.7)
        ax2.set_title(
            "Average Monthly Precipitation Pattern", fontsize=14, fontweight="bold"
        )
        ax2.set_xlabel("Month", fontsize=12)
        ax2.set_ylabel("Precipitation", fontsize=12)
        ax2.set_xticks(range(1, 13))
        ax2.set_xticklabels(months)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            f"{self.results_dir}/seasonal_patterns.png", dpi=300, bbox_inches="tight"
        )
        plt.show()

        self.analysis_results["seasonal_patterns"] = monthly_data

    def analyze_ghg_composition(self):
        """Analyze greenhouse gas composition trends"""
        print("\n🌍 Analyzing GHG Composition...")

        ghg_2010_2014 = self.datasets["ghg_2010_2014_enhanced"]
        ghg_2015_2020 = self.datasets["ghg_2015_2020_enhanced"]

        # Filter for main gas types
        gas_types = ["CO2", "CH4", "N2O"]

        # Extract average composition for each period
        composition_2010_2014 = {}
        composition_2015_2020 = {}

        for gas in gas_types:
            gas_data_early = ghg_2010_2014[ghg_2010_2014["Module"] == gas]
            gas_data_late = ghg_2015_2020[ghg_2015_2020["Module"] == gas]

            if not gas_data_early.empty:
                # Extract only numeric columns (years 2010-2014)
                numeric_cols = [
                    col
                    for col in gas_data_early.columns
                    if isinstance(col, (int, float)) and 2010 <= col <= 2014
                ]
                if numeric_cols:
                    composition_2010_2014[gas] = (
                        gas_data_early[numeric_cols].mean().mean()
                    )
            if not gas_data_late.empty:
                # Extract only numeric columns (years 2015-2020)
                numeric_cols = [
                    col
                    for col in gas_data_late.columns
                    if isinstance(col, (int, float)) and 2015 <= col <= 2020
                ]
                if numeric_cols:
                    composition_2015_2020[gas] = (
                        gas_data_late[numeric_cols].mean().mean()
                    )

        print("📊 GHG Composition Comparison:")
        comp_df = pd.DataFrame(
            {"2010-2014": composition_2010_2014, "2015-2020": composition_2015_2020}
        )
        print(comp_df.round(2))

        # Create composition visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Early period
        ax1.pie(
            composition_2010_2014.values(),
            labels=composition_2010_2014.keys(),
            autopct="%1.1f%%",
            startangle=90,
        )
        ax1.set_title("GHG Composition (2010-2014)", fontsize=14, fontweight="bold")

        # Late period
        ax2.pie(
            composition_2015_2020.values(),
            labels=composition_2015_2020.keys(),
            autopct="%1.1f%%",
            startangle=90,
        )
        ax2.set_title("GHG Composition (2015-2020)", fontsize=14, fontweight="bold")

        plt.tight_layout()
        plt.savefig(
            f"{self.results_dir}/ghg_composition.png", dpi=300, bbox_inches="tight"
        )
        plt.show()

        self.analysis_results["ghg_composition"] = comp_df

    def generate_eda_summary(self):
        """Generate comprehensive EDA summary"""
        print("\n" + "=" * 60)
        print("📋 EXPLORATORY DATA ANALYSIS SUMMARY")
        print("=" * 60)

        # Key findings
        print("\n🔍 KEY FINDINGS:")

        # Emissions trends
        if "emissions_growth" in self.analysis_results:
            max_growth_sector = max(
                self.analysis_results["emissions_growth"].items(), key=lambda x: x[1]
            )
            min_growth_sector = min(
                self.analysis_results["emissions_growth"].items(), key=lambda x: x[1]
            )
            print(
                f"📈 Fastest growing sector: {max_growth_sector[0]} ({max_growth_sector[1]:+.1f}%)"
            )
            print(
                f"📉 Slowest growing sector: {min_growth_sector[0]} ({min_growth_sector[1]:+.1f}%)"
            )

        # Sector contributions
        if "sector_contributions" in self.analysis_results:
            dominant_sector = self.analysis_results["sector_contributions"].idxmax()
            dominant_pct = self.analysis_results["sector_contributions"].max()
            print(
                f"🏭 Dominant sector: {dominant_sector.replace('_Pct', '')} ({dominant_pct:.1f}%)"
            )

        # Climate trends
        if "climate_trends" in self.analysis_results:
            temp_trend = self.analysis_results["climate_trends"]["temp_trend"]
            precip_trend = self.analysis_results["climate_trends"]["precip_trend"]
            print(f"🌡️ Temperature trend: {temp_trend:.4f}°C per year")
            print(f"🌧️ Precipitation trend: {precip_trend:.4f} units per year")

        # Strongest correlations
        if "correlations" in self.analysis_results:
            corr_with_temp = self.analysis_results["correlations"]["Temperature_Annual"]
            strongest_temp_corr = corr_with_temp.abs().idxmax()
            strongest_temp_value = corr_with_temp[strongest_temp_corr]
            print(
                f"🔗 Strongest temperature correlation: {strongest_temp_corr} (r = {strongest_temp_value:.3f})"
            )

            corr_with_precip = self.analysis_results["correlations"][
                "Precipitation_Annual"
            ]
            strongest_precip_corr = corr_with_precip.abs().idxmax()
            strongest_precip_value = corr_with_precip[strongest_precip_corr]
            print(
                f"💧 Strongest precipitation correlation: {strongest_precip_corr} (r = {strongest_precip_value:.3f})"
            )

        print("\n📊 Visualizations Created:")
        print(
            f"   - {self.results_dir}/emissions_trends.png: Sector-wise emission trends"
        )
        print(
            f"   - {self.results_dir}/sector_contributions.png: Pie chart of sector contributions"
        )
        print(
            f"   - {self.results_dir}/climate_trends.png: Temperature and precipitation trends"
        )
        print(f"   - {self.results_dir}/correlation_matrix.png: Correlation heatmap")
        print(
            f"   - {self.results_dir}/emissions_temperature_scatter.png: Scatter plots"
        )
        print(
            f"   - {self.results_dir}/seasonal_patterns.png: Monthly climate patterns"
        )
        print(
            f"   - {self.results_dir}/ghg_composition.png: GHG composition comparison"
        )

    def run_complete_eda(self):
        """Run complete exploratory data analysis"""
        print("🚀 Starting Comprehensive EmisGaz EDA")
        print("=" * 60)

        # Load data
        self.load_data()

        # Run all analyses
        self.analyze_emissions_trends()
        self.analyze_sector_contributions()
        self.analyze_climate_trends()
        self.analyze_correlations()
        self.analyze_seasonal_patterns()
        self.analyze_ghg_composition()

        # Generate summary
        self.generate_eda_summary()

        print("\n✅ EXPLORATORY DATA ANALYSIS COMPLETED!")
        print("=" * 60)


def main():
    """Main execution function"""
    eda = EmisGazEDA(results_dir="results")
    eda.run_complete_eda()


if __name__ == "__main__":
    main()
