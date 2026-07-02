import pytest
from fastapi.testclient import TestClient
from app.main import app

# Initialize the TestClient. This simulates a real user sending HTTP requests.
client = TestClient(app)

def test_health_endpoint():
    """Test that the health check returns 200 OK."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True

def test_predict_endpoint_success():
    """Test a valid prediction request."""
    # This is a valid payload matching our Pydantic schema
    valid_payload = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 300.0
    }
    
    response = client.post("/predict", json=valid_payload)
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "prediction" in data
    assert 0.0 <= data["churn_probability"] <= 1.0 # Probability must be between 0 and 1

def test_predict_endpoint_invalid_data():
    """Test that the API rejects invalid data (Pydantic validation)."""
    # Missing 'gender' and 'tenure' is an integer instead of string/number
    invalid_payload = {
        "SeniorCitizen": 0,
        "Partner": "Yes",
        # 'gender' is missing!
        "tenure": "twelve" # Wrong type!
    }
    
    response = client.post("/predict", json=invalid_payload)
    
    # FastAPI should automatically reject this with a 422 Unprocessable Entity
    assert response.status_code == 422 