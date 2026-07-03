import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import joblib
import wandb
import os


def main():
    # 1. Initialize W&B Run
    wandb.init(project="churn-prediction-mlops", name="rf-baseline-v1")

    # 2. Load Data (Telco Customer Churn)
    print("Loading data...")
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    df = pd.read_csv(url)

    # 3. Preprocessing
    print("Preprocessing data...")
    df.drop('customerID', axis=1, inplace=True)

    # FIX 1: Pandas Copy-on-Write fix (removed inplace=True)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

    # Target encoding
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # One-hot encode categorical variables
    df = pd.get_dummies(df, drop_first=True)

    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # 4. Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # FIX 2: Separate model hyperparameters from data splitting parameters
    model_params = {
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42
    }

    # Log all configuration to W&B
    wandb.config.update({
        "test_size": 0.2,
        **model_params
    })

    # 5. Train Model (Pass ONLY model_params here!)
    print("Training model...")
    model = RandomForestClassifier(**model_params)
    model.fit(X_train, y_train)

    # 6. Evaluate
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred_proba)
    }
    wandb.log(metrics)
    print("Metrics:", metrics)

    # 7. Save Model and Artifacts
    os.makedirs("models", exist_ok=True)

    # Save the model
    model_path = "models/random_forest_model.joblib"
    joblib.dump(model, model_path)

    # CRITICAL: Save the exact feature names the model expects!
    feature_names = X.columns.tolist()
    joblib.dump(feature_names, "models/feature_names.joblib")

    # Log model to W&B Artifacts
    artifact = wandb.Artifact(
        name="churn-random-forest",
        type="model",
        description="Random Forest model for Telco Churn prediction"
    )
    artifact.add_file(model_path)
    artifact.add_file("models/feature_names.joblib")
    wandb.log_artifact(artifact)

    print("Training complete! Check your W&B dashboard.")


if __name__ == "__main__":
    main()
