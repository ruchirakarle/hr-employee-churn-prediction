import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)
from imblearn.over_sampling import SMOTE
import joblib
import warnings

warnings.filterwarnings("ignore")


class ModelTrainer:
    """
    Train and evaluate multiple machine learning models for employee attrition prediction.

    This class implements a complete training pipeline including data loading,
    preprocessing, model training, evaluation, and comparison.
    """

    def __init__(self):
        """Initialize the ModelTrainer with empty containers for models and results."""
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None

    def load_data(self):
        """
        Load preprocessed and engineered data from CSV file.

        Returns:
            tuple: Feature matrix X and target vector y
        """
        print("\n" + "=" * 60)
        print("DATA LOADING")
        print("=" * 60)

        df = pd.read_csv("data/processed/engineered_data.csv")

        # Separate features and target variable
        X = df.drop("Attrition", axis=1)
        y = df["Attrition"]

        print(f"Dataset loaded successfully: {len(df)} samples")
        print(f"Number of features: {len(X.columns)}")
        print(f"Attrition rate: {y.mean()*100:.2f}%")

        return X, y

    def split_data(self, X, y, test_size=0.2):
        """
        Split data into training and testing sets with stratification.

        Args:
            X: Feature matrix
            y: Target vector
            test_size: Proportion of data to include in test split (default: 0.2)

        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        print("\n" + "=" * 60)
        print("DATA SPLITTING")
        print("=" * 60)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        print(f"Training set size: {len(X_train)} samples")
        print(f"  Positive class: {y_train.sum()} ({y_train.mean()*100:.2f}%)")
        print(f"Testing set size: {len(X_test)} samples")
        print(f"  Positive class: {y_test.sum()} ({y_test.mean()*100:.2f}%)")

        return X_train, X_test, y_train, y_test

    def handle_imbalance(self, X_train, y_train):
        """
        Apply Synthetic Minority Over-sampling Technique (SMOTE) to address class imbalance.

        Args:
            X_train: Training feature matrix
            y_train: Training target vector

        Returns:
            tuple: Balanced X_train and y_train
        """
        print("\n" + "=" * 60)
        print("CLASS IMBALANCE HANDLING - SMOTE")
        print("=" * 60)

        print(f"Class distribution before SMOTE:")
        print(f"  Class 0 (No Attrition): {(y_train == 0).sum()}")
        print(f"  Class 1 (Attrition): {(y_train == 1).sum()}")

        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

        print(f"\nClass distribution after SMOTE:")
        print(f"  Class 0 (No Attrition): {(y_train_balanced == 0).sum()}")
        print(f"  Class 1 (Attrition): {(y_train_balanced == 1).sum()}")
        print("Classes successfully balanced using SMOTE")

        return X_train_balanced, y_train_balanced

    def initialize_models(self):
        """
        Initialize ensemble of classification models with optimized hyperparameters.

        Models include:
        - Logistic Regression (baseline linear model)
        - Decision Tree (single tree model)
        - Random Forest (ensemble of trees)
        - Gradient Boosting (sequential ensemble)
        - XGBoost (optimized gradient boosting)
        """
        print("\n" + "=" * 60)
        print("MODEL INITIALIZATION")
        print("=" * 60)

        self.models = {
            "Logistic Regression": LogisticRegression(
                random_state=42, max_iter=1000, class_weight="balanced"
            ),
            "Decision Tree": DecisionTreeClassifier(
                random_state=42,
                max_depth=10,
                min_samples_split=20,
                class_weight="balanced",
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10,
                min_samples_split=20,
                class_weight="balanced",
                n_jobs=-1,
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
            ),
            "XGBoost": XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42,
                scale_pos_weight=3,
                eval_metric="logloss",
            ),
        }

        print(f"Initialized {len(self.models)} classification models:")
        for i, name in enumerate(self.models.keys(), 1):
            print(f"  {i}. {name}")

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """
        Train all models and evaluate their performance on test set.

        Args:
            X_train: Training features
            X_test: Testing features
            y_train: Training labels
            y_test: Testing labels
        """
        print("\n" + "=" * 60)
        print("MODEL TRAINING AND EVALUATION")
        print("=" * 60)

        for name, model in self.models.items():
            print(f"\n{'='*60}")
            print(f"Training: {name}")
            print(f"{'='*60}")

            # Model training
            model.fit(X_train, y_train)

            # Generate predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            # Calculate performance metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)

            # Store results for comparison
            self.results[name] = {
                "model": model,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "roc_auc": roc_auc,
                "y_pred": y_pred,
                "y_pred_proba": y_pred_proba,
            }

            # Display performance metrics
            print(f"\nPerformance Metrics:")
            print(f"  Accuracy:  {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1-Score:  {f1:.4f}")
            print(f"  ROC-AUC:   {roc_auc:.4f}")

        print("\n" + "=" * 60)
        print("MODEL TRAINING COMPLETED")
        print("=" * 60)

    def compare_models(self):
        """
        Compare performance of all trained models and select the best performer.

        Returns:
            DataFrame: Comparison of all models across evaluation metrics
        """
        print("\n" + "=" * 60)
        print("MODEL PERFORMANCE COMPARISON")
        print("=" * 60)

        # Create comparison dataframe
        comparison_df = pd.DataFrame(
            {
                "Model": list(self.results.keys()),
                "Accuracy": [self.results[m]["accuracy"] for m in self.results],
                "Precision": [self.results[m]["precision"] for m in self.results],
                "Recall": [self.results[m]["recall"] for m in self.results],
                "F1-Score": [self.results[m]["f1"] for m in self.results],
                "ROC-AUC": [self.results[m]["roc_auc"] for m in self.results],
            }
        )

        # Sort by F1-Score (balanced metric for imbalanced datasets)
        comparison_df = comparison_df.sort_values("F1-Score", ascending=False)

        print("\nComparative Performance Analysis:")
        print(comparison_df.to_string(index=False))

        # Select best performing model based on F1-Score
        self.best_model_name = comparison_df.iloc[0]["Model"]
        self.best_model = self.results[self.best_model_name]["model"]

        print(f"\nBest Performing Model: {self.best_model_name}")
        print(f"  F1-Score: {comparison_df.iloc[0]['F1-Score']:.4f}")
        print(f"  ROC-AUC:  {comparison_df.iloc[0]['ROC-AUC']:.4f}")

        return comparison_df

    def plot_comparison(self, comparison_df):
        """
        Generate visualization comparing model performance across metrics.

        Args:
            comparison_df: DataFrame containing model comparison results
        """
        print("\nGenerating model comparison visualization...")

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Plot 1: Multi-metric comparison across all models
        metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
        x = np.arange(len(self.results))
        width = 0.15

        # FIXED: Create proper mapping for metric names to dictionary keys
        metric_map = {
            "Accuracy": "accuracy",
            "Precision": "precision",
            "Recall": "recall",
            "F1-Score": "f1",  # Key fix: maps to 'f1', not 'f1_score'
            "ROC-AUC": "roc_auc",
        }

        for i, metric in enumerate(metrics):
            metric_key = metric_map[metric]
            values = [self.results[m][metric_key] for m in self.results]
            axes[0].bar(x + i * width, values, width, label=metric)

        axes[0].set_xlabel("Models")
        axes[0].set_ylabel("Score")
        axes[0].set_title("Model Performance Comparison - All Metrics")
        axes[0].set_xticks(x + width * 2)
        axes[0].set_xticklabels(self.results.keys(), rotation=45, ha="right")
        axes[0].legend()
        axes[0].grid(axis="y", alpha=0.3)

        # Plot 2: F1-Score focused comparison
        colors = [
            "#2ecc71" if m == self.best_model_name else "#3498db"
            for m in comparison_df["Model"]
        ]
        axes[1].barh(comparison_df["Model"], comparison_df["F1-Score"], color=colors)
        axes[1].set_xlabel("F1-Score")
        axes[1].set_title(f"F1-Score Comparison (Best: {self.best_model_name})")
        axes[1].grid(axis="x", alpha=0.3)

        plt.tight_layout()
        plt.savefig("assets/model_comparison.png", dpi=300, bbox_inches="tight")
        print("Saved: assets/model_comparison.png")
        plt.close()

    def plot_confusion_matrix(self, y_test):
        """
        Generate confusion matrix visualization for best performing model.

        Args:
            y_test: True labels from test set
        """
        print("\nGenerating confusion matrix...")

        y_pred = self.results[self.best_model_name]["y_pred"]
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["No", "Yes"],
            yticklabels=["No", "Yes"],
        )
        plt.title(f"Confusion Matrix - {self.best_model_name}")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")

        # Add percentage annotations
        total = cm.sum()
        for i in range(2):
            for j in range(2):
                plt.text(
                    j + 0.5,
                    i + 0.7,
                    f"({cm[i,j]/total*100:.1f}%)",
                    ha="center",
                    va="center",
                    color="red",
                    fontsize=10,
                )

        plt.tight_layout()
        plt.savefig("assets/confusion_matrix.png", dpi=300, bbox_inches="tight")
        print("Saved: assets/confusion_matrix.png")
        plt.close()

    def plot_roc_curve(self, y_test):
        """
        Generate ROC curve for best performing model.

        Args:
            y_test: True labels from test set
        """
        print("\nGenerating ROC curve...")

        y_pred_proba = self.results[self.best_model_name]["y_pred_proba"]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = self.results[self.best_model_name]["roc_auc"]

        plt.figure(figsize=(8, 6))
        plt.plot(
            fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})"
        )
        plt.plot(
            [0, 1],
            [0, 1],
            color="navy",
            lw=2,
            linestyle="--",
            label="Random Classifier",
        )
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(
            f"Receiver Operating Characteristic (ROC) Curve - {self.best_model_name}"
        )
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig("assets/roc_curve.png", dpi=300, bbox_inches="tight")
        print("Saved: assets/roc_curve.png")
        plt.close()

    def plot_feature_importance(self, feature_names):
        """
        Generate feature importance plot for tree-based models.

        Args:
            feature_names: List of feature names
        """
        print("\nGenerating feature importance plot...")

        # Feature importance is only available for tree-based models
        if hasattr(self.best_model, "feature_importances_"):
            importances = self.best_model.feature_importances_

            # Create dataframe and sort by importance
            feat_imp_df = (
                pd.DataFrame({"Feature": feature_names, "Importance": importances})
                .sort_values("Importance", ascending=False)
                .head(15)
            )

            # Generate horizontal bar plot
            plt.figure(figsize=(10, 8))
            plt.barh(feat_imp_df["Feature"], feat_imp_df["Importance"], color="#3498db")
            plt.xlabel("Importance Score")
            plt.title(f"Top 15 Feature Importance - {self.best_model_name}")
            plt.gca().invert_yaxis()
            plt.grid(axis="x", alpha=0.3)

            plt.tight_layout()
            plt.savefig("assets/feature_importance.png", dpi=300, bbox_inches="tight")
            print("Saved: assets/feature_importance.png")
            plt.close()
        else:
            print("Note: Feature importance not available for this model type")

    def save_best_model(self):
        """Save the best performing model and its metadata to disk."""
        print("\n" + "=" * 60)
        print("MODEL PERSISTENCE")
        print("=" * 60)

        # Save trained model
        joblib.dump(self.best_model, "models/best_model.pkl")

        # Save model metadata
        model_info = {
            "model_name": self.best_model_name,
            "accuracy": self.results[self.best_model_name]["accuracy"],
            "precision": self.results[self.best_model_name]["precision"],
            "recall": self.results[self.best_model_name]["recall"],
            "f1_score": self.results[self.best_model_name]["f1"],
            "roc_auc": self.results[self.best_model_name]["roc_auc"],
        }
        joblib.dump(model_info, "models/model_info.pkl")

        print(f"Best model saved: {self.best_model_name}")
        print(f"Location: models/best_model.pkl")
        print(f"Model metadata saved: models/model_info.pkl")

    def print_classification_report(self, y_test):
        """
        Print detailed classification report for best model.

        Args:
            y_test: True labels from test set
        """
        print("\n" + "=" * 60)
        print(f"CLASSIFICATION REPORT - {self.best_model_name}")
        print("=" * 60)

        y_pred = self.results[self.best_model_name]["y_pred"]
        print(
            classification_report(
                y_test, y_pred, target_names=["No Attrition", "Attrition"]
            )
        )


def main():
    """
    Main training pipeline execution.

    This function orchestrates the complete model training workflow including:
    1. Data loading
    2. Train-test splitting
    3. Class imbalance handling
    4. Model initialization
    5. Training and evaluation
    6. Model comparison
    7. Visualization generation
    8. Model persistence
    """
    print("\n" + "=" * 60)
    print("HR EMPLOYEE ATTRITION PREDICTION - MODEL TRAINING PIPELINE")
    print("=" * 60)

    # Initialize trainer object
    trainer = ModelTrainer()

    # Step 1: Load preprocessed and engineered data
    X, y = trainer.load_data()

    # Step 2: Split data into training and testing sets
    X_train, X_test, y_train, y_test = trainer.split_data(X, y)

    # Step 3: Handle class imbalance using SMOTE
    X_train_balanced, y_train_balanced = trainer.handle_imbalance(X_train, y_train)

    # Step 4: Initialize classification models
    trainer.initialize_models()

    # Step 5: Train and evaluate all models
    trainer.train_and_evaluate(X_train_balanced, X_test, y_train_balanced, y_test)

    # Step 6: Compare model performances
    comparison_df = trainer.compare_models()

    # Step 7: Generate performance visualizations
    trainer.plot_comparison(comparison_df)
    trainer.plot_confusion_matrix(y_test)
    trainer.plot_roc_curve(y_test)
    trainer.plot_feature_importance(X.columns.tolist())

    # Step 8: Print detailed classification metrics
    trainer.print_classification_report(y_test)

    # Step 9: Persist best model to disk
    trainer.save_best_model()

    print("\n" + "=" * 60)
    print("MODEL TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()
