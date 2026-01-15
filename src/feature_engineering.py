import pandas as pd
import numpy as np


class FeatureEngineer:
    """
    Create simple, robust features
    """

    def __init__(self):
        self.feature_descriptions = {}

    def create_all_features(self, df):
        """Create only essential features"""
        print("\n" + "=" * 50)
        print("FEATURE ENGINEERING (SIMPLE VERSION)")
        print("=" * 50)

        df = df.copy()
        initial_features = len(df.columns)

        # Just 5 simple, robust features
        df = self.create_simple_features(df)

        final_features = len(df.columns)
        print(f"\n[SUCCESS] Created {final_features - initial_features} new features")
        print(f"Total features: {final_features}")

        return df

    def create_simple_features(self, df):
        """Create simple, robust features that won't break"""
        print("\n[STEP] Creating simple features...")

        # 1. Total Satisfaction (sum of satisfaction scores)
        satisfaction_cols = [
            "JobSatisfaction",
            "EnvironmentSatisfaction",
            "RelationshipSatisfaction",
        ]
        available_cols = [col for col in satisfaction_cols if col in df.columns]
        if len(available_cols) >= 2:
            df["TotalSatisfaction"] = df[available_cols].sum(axis=1)
            self.feature_descriptions["TotalSatisfaction"] = "Sum of satisfaction scores"

        # 2. Work-Life Score (overtime penalty)
        if "OverTime" in df.columns and "WorkLifeBalance" in df.columns:
            df["WorkLifeScore"] = df["WorkLifeBalance"] - (2 * df["OverTime"])
            self.feature_descriptions["WorkLifeScore"] = "Work-life balance - overtime penalty"

        # 3. Engagement Score (involvement × satisfaction)
        if "JobInvolvement" in df.columns and "JobSatisfaction" in df.columns:
            df["EngagementScore"] = df["JobInvolvement"] * df["JobSatisfaction"]
            self.feature_descriptions["EngagementScore"] = "Job involvement x satisfaction"

        # 4. Overtime Distance Stress (overtime × commute)
        if "OverTime" in df.columns and "DistanceFromHome" in df.columns:
            df["OvertimeDistanceStress"] = df["OverTime"] * df["DistanceFromHome"]
            self.feature_descriptions["OvertimeDistanceStress"] = "Overtime x distance"

        # 5. Compensation Package (stock × job level)
        if "StockOptionLevel" in df.columns and "JobLevel" in df.columns:
            df["CompensationPackage"] = df["StockOptionLevel"] * df["JobLevel"]
            self.feature_descriptions["CompensationPackage"] = "Stock options x job level"

        print(f"  -> Created {len(self.feature_descriptions)} features")
        return df

    def print_feature_summary(self):
        """Print created features"""
        print("\n" + "=" * 50)
        print("ENGINEERED FEATURES SUMMARY")
        print("=" * 50)
        for feature, description in self.feature_descriptions.items():
            print(f"{feature:30} | {description}")


# Example usage
if __name__ == "__main__":
    # Load preprocessed data
    df = pd.read_csv('data/processed/processed_data.csv')
    
    # Separate features and target
    y = df['Attrition']
    X = df.drop(columns=['Attrition'])
    
    # Engineer features
    engineer = FeatureEngineer()
    X_engineered = engineer.create_all_features(X)
    
    # Print summary
    engineer.print_feature_summary()
    
    # Save
    X_engineered['Attrition'] = y
    X_engineered.to_csv('data/processed/engineered_data.csv', index=False)
    print("\n[SUCCESS] Saved engineered data to data/processed/engineered_data.csv")