"""
EmisGaz Data Validation and Quality Check Script
Comprehensive validation of prepared data for emissions and climate analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings("ignore")


class EmisGazDataValidator:
    """
    Data validation class for EmisGaz project
    Performs comprehensive quality checks on prepared datasets
    """

    def __init__(self, prepared_data_path: str = "prepared_data.xlsx"):
        self.prepared_data_path = prepared_data_path
        self.validation_results = {}
        self.quality_issues = []

    def load_prepared_data(self) -> Dict[str, pd.DataFrame]:
        """Load all prepared datasets from Excel file"""
        print("📂 Loading prepared data for validation...")

        try:
            excel_file = pd.ExcelFile(self.prepared_data_path)
            prepared_data = {}

            for sheet_name in excel_file.sheet_names:
                prepared_data[sheet_name] = pd.read_excel(
                    self.prepared_data_path, sheet_name=sheet_name, index_col=0
                )

            print(f"✅ Loaded {len(prepared_data)} datasets")
            return prepared_data

        except Exception as e:
            print(f"❌ Error loading prepared data: {e}")
            return {}

    def validate_data_types(self, df: pd.DataFrame, dataset_name: str) -> List[str]:
        """Validate data types and numeric consistency"""
        issues = []

        # Check for non-numeric values in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns

        if len(non_numeric_cols) > 0:
            issues.append(f"Non-numeric columns found: {list(non_numeric_cols)}")

        # Check for infinite values
        for col in numeric_cols:
            if np.any(np.isinf(df[col])):
                issues.append(f"Infinite values found in {col}")

        return issues

    def validate_missing_values(self, df: pd.DataFrame, dataset_name: str) -> List[str]:
        """Validate missing values and data completeness"""
        issues = []

        total_missing = df.isnull().sum().sum()
        if total_missing > 0:
            missing_by_col = df.isnull().sum()
            missing_cols = missing_by_col[missing_by_col > 0]

            issues.append(f"Missing values: {total_missing} total")
            for col, count in missing_cols.items():
                pct_missing = (count / len(df)) * 100
                issues.append(f"  - {col}: {count} missing ({pct_missing:.1f}%)")

        return issues

    def validate_value_ranges(self, df: pd.DataFrame, dataset_name: str) -> List[str]:
        """Validate value ranges for different data types"""
        issues = []

        if "emissions" in dataset_name:
            # Emissions should be positive
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if (df[col] < 0).any():
                    negative_count = (df[col] < 0).sum()
                    issues.append(
                        f"Negative values in {col}: {negative_count} instances"
                    )

        elif "temperature" in dataset_name:
            # Temperature should be in reasonable range (e.g., -50 to 50°C)
            if "ANN" in df.columns:
                if (df["ANN"] < -50).any() or (df["ANN"] > 50).any():
                    issues.append(
                        "Temperature values outside reasonable range (-50 to 50°C)"
                    )

        elif "precipitation" in dataset_name:
            # Precipitation should be non-negative
            if "ANN" in df.columns:
                if (df["ANN"] < 0).any():
                    issues.append("Negative precipitation values found")

        return issues

    def validate_time_series_consistency(
        self, df: pd.DataFrame, dataset_name: str
    ) -> List[str]:
        """Validate time series data consistency"""
        issues = []

        if "timeseries" in dataset_name or "emissions" in dataset_name:
            # Check for consistent time intervals
            if hasattr(df.index, "dtype") and np.issubdtype(df.index.dtype, np.number):
                years = df.index
                if len(years) > 1:
                    year_diffs = np.diff(years)
                    if not all(diff == 1 for diff in year_diffs):
                        issues.append("Non-consecutive years in time series")

        return issues

    def validate_dataset_structure(
        self, df: pd.DataFrame, dataset_name: str
    ) -> List[str]:
        """Validate dataset structure and dimensions"""
        issues = []

        # Check for empty dataset
        if df.empty:
            issues.append("Dataset is empty")
            return issues

        # Check for minimum required rows/columns
        if df.shape[0] < 1:
            issues.append("Dataset has no rows")

        if df.shape[1] < 1:
            issues.append("Dataset has no columns")

        # Check for duplicate rows
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            issues.append(f"Duplicate rows found: {duplicates}")

        # Check for duplicate column names
        if len(df.columns) != len(set(df.columns)):
            issues.append("Duplicate column names found")

        return issues

    def validate_emissions_specific(
        self, df: pd.DataFrame, dataset_name: str
    ) -> List[str]:
        """Validate emissions-specific business rules"""
        issues = []

        if "emissions" in dataset_name:
            # Check for expected sectors
            expected_sectors = ["Energie", "Agriculture", "PIUP", "Déchets"]

            if "Sector" in df.columns or df.index.name == "Sector":
                sectors = df.index if df.index.name == "Sector" else df["Sector"]

                for sector in expected_sectors:
                    if sector not in sectors:
                        issues.append(f"Expected sector '{sector}' not found")

            # Check for reasonable emission values (not too large/small)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].max() > 1e6:  # Arbitrary large number
                    issues.append(f"Unusually large emission value in {col}")
                if df[col].min() < -1e6:  # Arbitrary small number
                    issues.append(f"Unusually small emission value in {col}")

        return issues

    def validate_climate_specific(
        self, df: pd.DataFrame, dataset_name: str
    ) -> List[str]:
        """Validate climate-specific business rules"""
        issues = []

        if "temperature" in dataset_name or "precipitation" in dataset_name:
            # Check for expected month columns
            expected_months = [
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

            missing_months = [
                month for month in expected_months if month not in df.columns
            ]
            if missing_months:
                issues.append(f"Missing month columns: {missing_months}")

        return issues

    def run_comprehensive_validation(self) -> Dict[str, Dict]:
        """Run comprehensive validation on all prepared datasets"""
        print("🔍 Starting comprehensive data validation...")
        print("=" * 60)

        prepared_data = self.load_prepared_data()

        if not prepared_data:
            print("❌ No data to validate")
            return {}

        validation_summary = {}

        for dataset_name, df in prepared_data.items():
            print(f"\n📊 Validating: {dataset_name} ({df.shape})")

            dataset_issues = []

            # Run all validation checks
            dataset_issues.extend(self.validate_dataset_structure(df, dataset_name))
            dataset_issues.extend(self.validate_data_types(df, dataset_name))
            dataset_issues.extend(self.validate_missing_values(df, dataset_name))
            dataset_issues.extend(self.validate_value_ranges(df, dataset_name))
            dataset_issues.extend(
                self.validate_time_series_consistency(df, dataset_name)
            )
            dataset_issues.extend(self.validate_emissions_specific(df, dataset_name))
            dataset_issues.extend(self.validate_climate_specific(df, dataset_name))

            # Store results
            validation_summary[dataset_name] = {
                "shape": df.shape,
                "total_issues": len(dataset_issues),
                "issues": dataset_issues,
                "status": "PASS" if len(dataset_issues) == 0 else "ISSUES",
            }

            # Print results for this dataset
            if len(dataset_issues) == 0:
                print("  ✅ PASS - No issues found")
            else:
                print(f"  ⚠️  ISSUES - {len(dataset_issues)} issues found")
                for issue in dataset_issues[:3]:  # Show first 3 issues
                    print(f"    - {issue}")
                if len(dataset_issues) > 3:
                    print(f"    ... and {len(dataset_issues) - 3} more issues")

        self.validation_results = validation_summary
        return validation_summary

    def generate_validation_report(self) -> str:
        """Generate a comprehensive validation report"""
        if not self.validation_results:
            print("❌ No validation results available. Run validation first.")
            return ""

        print("\n" + "=" * 60)
        print("📋 VALIDATION REPORT SUMMARY")
        print("=" * 60)

        total_datasets = len(self.validation_results)
        datasets_with_issues = sum(
            1
            for result in self.validation_results.values()
            if result["status"] == "ISSUES"
        )
        total_issues = sum(
            result["total_issues"] for result in self.validation_results.values()
        )

        print(f"📊 Datasets validated: {total_datasets}")
        print(f"⚠️  Datasets with issues: {datasets_with_issues}")
        print(f"🔧 Total issues found: {total_issues}")

        # Detailed report
        report_lines = []
        report_lines.append("EMISGAZ DATA VALIDATION REPORT")
        report_lines.append("=" * 50)
        report_lines.append(
            f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        report_lines.append(f"Total datasets: {total_datasets}")
        report_lines.append(f"Datasets with issues: {datasets_with_issues}")
        report_lines.append(f"Total issues: {total_issues}")
        report_lines.append("")

        for dataset_name, result in self.validation_results.items():
            report_lines.append(f"DATASET: {dataset_name}")
            report_lines.append(f"  Shape: {result['shape']}")
            report_lines.append(f"  Status: {result['status']}")
            report_lines.append(f"  Issues: {result['total_issues']}")

            if result["issues"]:
                report_lines.append("  Details:")
                for issue in result["issues"]:
                    report_lines.append(f"    - {issue}")
            report_lines.append("")

        # Overall assessment
        if total_issues == 0:
            report_lines.append("🎉 OVERALL ASSESSMENT: EXCELLENT")
            report_lines.append("All datasets passed validation checks successfully!")
        elif total_issues < 10:
            report_lines.append("✅ OVERALL ASSESSMENT: GOOD")
            report_lines.append("Minor issues found that can be addressed.")
        else:
            report_lines.append("⚠️  OVERALL ASSESSMENT: NEEDS ATTENTION")
            report_lines.append("Significant issues found that require attention.")

        full_report = "\n".join(report_lines)

        # Save report to file
        report_filename = "data_validation_report.txt"
        with open(report_filename, "w", encoding="utf-8") as f:
            f.write(full_report)

        print(f"📄 Validation report saved to: {report_filename}")
        return full_report

    def get_data_quality_score(self) -> float:
        """Calculate overall data quality score (0-100)"""
        if not self.validation_results:
            return 0.0

        total_penalty = 0
        max_penalty_per_dataset = 10  # Maximum penalty per dataset

        for result in self.validation_results.values():
            # Penalty based on number of issues (capped at max_penalty_per_dataset)
            penalty = min(result["total_issues"], max_penalty_per_dataset)
            total_penalty += penalty

        # Calculate score (100 - penalty percentage)
        max_possible_penalty = len(self.validation_results) * max_penalty_per_dataset
        quality_score = max(0, 100 - (total_penalty / max_possible_penalty * 100))

        return round(quality_score, 1)


def main():
    """Main execution function for data validation"""
    print("🚀 Starting EmisGaz Data Validation Process")
    print("=" * 60)

    # Initialize validator
    validator = EmisGazDataValidator()

    # Run comprehensive validation
    validation_results = validator.run_comprehensive_validation()

    if not validation_results:
        print("❌ Validation failed - no results generated")
        return

    # Generate and display report
    report = validator.generate_validation_report()

    # Calculate quality score
    quality_score = validator.get_data_quality_score()
    print(f"\n🏆 Overall Data Quality Score: {quality_score}/100")

    if quality_score >= 90:
        print("🎉 Excellent data quality! Ready for analysis.")
    elif quality_score >= 70:
        print("✅ Good data quality. Minor issues can be addressed.")
    else:
        print("⚠️  Data quality needs improvement before analysis.")


if __name__ == "__main__":
    main()
