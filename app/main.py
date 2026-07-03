from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from app.schemas import CustomerData

# 1. Initialize FastAPI App
app = FastAPI(
    title="Churn Prediction API",
    description="An MLOps production API for predicting customer churn.",
    version="1.0.0"
)

# 2. Load Model and Artifacts into Memory
print("Loading model and artifacts...")
model = joblib.load("models/random_forest_model.joblib")
feature_names = joblib.load("models/feature_names.joblib")
print("Model loaded successfully!")


# 3. The Preprocessing Function
def preprocess_input(data: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])

    # Handle TotalCharges missing values.
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0.0)

    # One-hot encode the categorical variables
    df = pd.get_dummies(df, drop_first=True)

    # THE MOST CRITICAL STEP: Feature Alignment
    df = df.reindex(columns=feature_names, fill_value=0)

    return df


# 4. Endpoints

@app.get("/health", tags=["Monitoring"])
def health_check():
    """
    Standard endpoint used by cloud providers (like Render or AWS)
    to check if the container is alive and ready to take traffic.
    """
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", tags=["Inference"])
def predict_churn(customer: CustomerData):
    try:
        # Convert Pydantic model to dictionary
        raw_data = customer.model_dump()

        # Preprocess the data
        processed_df = preprocess_input(raw_data)

        # Predict
        prediction = model.predict(processed_df)[0]
        probabilities = model.predict_proba(processed_df)[0]

        return {
            "prediction": int(prediction),
            "churn_probability": float(probabilities[1]),
            "status": "success"
        }

    except Exception as e:
        # In production, we log the error to a monitoring system,
        # but we return a generic 500 error to the user for security.
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
