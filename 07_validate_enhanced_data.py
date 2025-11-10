"""
Quick validation script for enhanced EmisGaz data
"""

import pandas as pd
import numpy as np


def quick_validate_enhanced_data():
    """Quick validation of enhanced data quality"""
    print("🔍 Quick Validation of Enhanced EmisGaz Data")
    print("=" * 50)

    try:
        # Load enhanced data
        enhanced_data = pd.read_excel("enhanced_data.xlsx", sheet_name=None)

        print(f"📊 Loaded {len(enhanced_data)} enhanced datasets")

        total_issues = 0
        datasets_with_issues = 0

        for sheet_name, df in enhanced_data.items():
            print(f"\n📋 {sheet_name}: {df.shape}")

            issues = []

            # Check for missing values
            missing_total = df.isnull().sum().sum()
            if missing_total > 0:
                issues.append(f"Missing values: {missing_total}")

            # Check for negative emissions (should not exist after cleaning)
            if "emissions" in sheet_name.lower():
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if (df[col] < 0).any():
                        negative_count = (df[col] < 0).sum()
                        issues.append(f"Negative values in {col}: {negative_count}")

            # Check for infinite values
            for col in df.select_dtypes(include=[np.number]).columns:
                if np.any(np.isinf(df[col])):
                    issues.append(f"Infinite values in {col}")

            # Report results
            if issues:
                datasets_with_issues += 1
                total_issues += len(issues)
                print(f"  ⚠️  {len(issues)} issues:")
                for issue in issues[:3]:  # Show first 3 issues
                    print(f"    - {issue}")
            else:
                print("  ✅ No issues found")

        # Calculate quality score
        quality_score = max(0, 100 - (total_issues * 2))  # Simple scoring
        quality_score = min(100, quality_score)

        print("\n" + "=" * 50)
        print("📋 VALIDATION SUMMARY")
        print(f"Total datasets: {len(enhanced_data)}")
        print(f"Datasets with issues: {datasets_with_issues}")
        print(f"Total issues found: {total_issues}")
        print(f"🏆 Quality Score: {quality_score:.1f}/100")

        if quality_score >= 90:
            print("🎉 Excellent data quality! Ready for analysis.")
        elif quality_score >= 70:
            print("✅ Good data quality. Ready for analysis.")
        else:
            print("⚠️  Data quality needs attention.")

    except Exception as e:
        print(f"❌ Error during validation: {e}")


if __name__ == "__main__":
    quick_validate_enhanced_data()
