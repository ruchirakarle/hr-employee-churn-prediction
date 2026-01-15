import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os


class DataPreprocessor:
    """
    Handles all data preprocessing for HR attrition prediction
    """

    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = None

    def load_data(self, filepath):
        """Load raw data"""
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        print(f" Loaded {len(df)} rows, {len(df.columns)} columns")
        return df

    def remove_unnecessary_columns(self, df):
        """Remove columns that won't help prediction"""
        # These columns have same value for all rows or are identifiers
        cols_to_drop = []

        # Check for columns with single unique value
        for col in df.columns:
            if df[col].nunique() == 1:
                cols_to_drop.append(col)
                print(f" Dropping {col} - only one unique value: {df[col].unique()[0]}")

        # Drop EmployeeNumber (just an ID)
        if "EmployeeNumber" in df.columns:
            cols_to_drop.append("EmployeeNumber")
            print(" Dropping EmployeeNumber - just an identifier")

        # Drop Over18 (everyone is over 18)
        if "Over18" in df.columns:
            cols_to_drop.append("Over18")

        # Drop EmployeeCount (all values are 1)
        if "EmployeeCount" in df.columns:
            cols_to_drop.append("EmployeeCount")

        # Drop StandardHours (all values are 80)
        if "StandardHours" in df.columns:
            cols_to_drop.append("StandardHours")

        if cols_to_drop:
            df = df.drop(columns=cols_to_drop, errors="ignore")
            print(f" Dropped {len(cols_to_drop)} unnecessary columns")

        return df

    def encode_target(self, df, target_col="Attrition"):
        """Encode target variable"""
        df[target_col] = df[target_col].map({"Yes": 1, "No": 0})
        print(f" Encoded target: Yes→1, No→0")
        return df

    def encode_categorical_features(self, df, fit=True):
        """Encode categorical variables"""
        categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

        # Remove target if present
        if "Attrition" in categorical_cols:
            categorical_cols.remove("Attrition")

        print(f"\nEncoding {len(categorical_cols)} categorical features...")

        for col in categorical_cols:
            if fit:
                # Create and fit new encoder
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                # Use existing encoder
                if col in self.label_encoders:
                    df[col] = self.label_encoders[col].transform(df[col].astype(str))
                else:
                    print(f"⚠️ Warning: No encoder found for {col}")

            print(f"  - {col}: {len(df[col].unique())} categories")

        print(" Categorical encoding complete")
        return df

    def scale_numerical_features(self, df, fit=True):
        """Scale numerical features"""
        # Identify numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Remove target and already binary columns
        exclude_cols = ["Attrition"]
        binary_cols = [col for col in numerical_cols if df[col].nunique() <= 2]

        numerical_cols = [
            col
            for col in numerical_cols
            if col not in exclude_cols and col not in binary_cols
        ]

        print(f"\nScaling {len(numerical_cols)} numerical features...")

        if fit:
            df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
        else:
            df[numerical_cols] = self.scaler.transform(df[numerical_cols])

        print(" Numerical scaling complete")
        return df

    def save_preprocessor(self, save_dir="models"):
        """Save label encoders and scaler"""
        os.makedirs(save_dir, exist_ok=True)

        joblib.dump(self.label_encoders, f"{save_dir}/label_encoders.pkl")
        joblib.dump(self.scaler, f"{save_dir}/scaler.pkl")
        joblib.dump(self.feature_names, f"{save_dir}/feature_names.pkl")

        print(f"\n Saved preprocessor to {save_dir}/")

    def load_preprocessor(self, load_dir="models"):
        """Load label encoders and scaler"""
        self.label_encoders = joblib.load(f"{load_dir}/label_encoders.pkl")
        self.scaler = joblib.load(f"{load_dir}/scaler.pkl")
        self.feature_names = joblib.load(f"{load_dir}/feature_names.pkl")

        print(f" Loaded preprocessor from {load_dir}/")

    def preprocess_pipeline(self, df, fit=True, target_col="Attrition"):
        """Complete preprocessing pipeline"""
        print("\n" + "=" * 50)
        print("STARTING PREPROCESSING PIPELINE")
        print("=" * 50)

        # Remove unnecessary columns
        df = self.remove_unnecessary_columns(df)

        # Separate features and target
        if target_col in df.columns:
            y = df[target_col]
            X = df.drop(columns=[target_col])
        else:
            y = None
            X = df

        # Encode target
        if y is not None:
            y = y.map({"Yes": 1, "No": 0}) if y.dtype == "object" else y

        # Encode categorical
        X = self.encode_categorical_features(X, fit=fit)

        # Scale numerical
        X = self.scale_numerical_features(X, fit=fit)

        # Store feature names
        if fit:
            self.feature_names = X.columns.tolist()

        print("\n" + "=" * 50)
        print(" PREPROCESSING COMPLETE")
        print("=" * 50)
        print(f"Features: {len(X.columns)}")
        print(f"Samples: {len(X)}")

        if y is not None:
            return X, y
        return X


# Example usage
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = DataPreprocessor()

    # Load data
    df = preprocessor.load_data("data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv")

    # Preprocess
    X, y = preprocessor.preprocess_pipeline(df, fit=True)

    # Save
    preprocessor.save_preprocessor()

    # Save processed data
    processed_df = X.copy()
    processed_df["Attrition"] = y
    processed_df.to_csv("data/processed/processed_data.csv", index=False)
    print("\n Saved processed data to data/processed/")
