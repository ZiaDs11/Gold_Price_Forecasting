import os
from typing import List, Any 

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, HTTPException, Depends, status
from pydantic import BaseModel, field_validator

# --- 1. Configuration & Global State ---
MODEL_PATH = os.getenv("MODEL_PATH", "models/gold_lstm_model.h5")
SCALER_PATH = os.getenv("SCALER_PATH", "models/scaler.save")
CSV_PATH = os.getenv("CSV_PATH", "data/goldstock.csv")
TIMESTEP = 60 

_MODEL = None
_SCALER = None


# --- 2. Pydantic Schemas ---
class PredictRequest(BaseModel):
    """Schema for the /predict endpoint request body."""
    recent_closes: List[float]
    days: int = 1

    @field_validator('recent_closes')
    def validate_closes_length(cls, v):
        if len(v) < TIMESTEP:
            raise ValueError(f"recent_closes must contain at least {TIMESTEP} values for the model.")
        return v

class ForecastRequest(BaseModel):
    """Schema for the /forecast endpoint request body."""
    days: int = 30
    
class PredictionResponse(BaseModel):
    """Schema for the standard prediction/forecast response."""
    results: List[float]


# --- 3. Business Logic / Service Layer ---
class GoldPredictorService:
    """
    Handles all prediction and forecasting logic.
    """
    # FIX APPLIED HERE: Using Any for the scaler type hint to avoid the AttributeError
    def __init__(self, model: tf.keras.Model, scaler: Any, timestep: int):
        self.model = model
        self.scaler = scaler
        self.timestep = timestep

    def _update_input_window(self, Xin: np.ndarray, new_val: np.ndarray) -> np.ndarray:
        """Helper to shift the input window and insert the new prediction."""
        Xin[:, :-1, :] = Xin[:, 1:, :] 
        Xin[:, -1, :] = new_val        
        return Xin

    def predict_n_days(self, recent_closes: List[float], days: int) -> List[float]:
        """Generates predictions for 'days' using the provided 'recent_closes' window."""
        window = np.array(recent_closes[-self.timestep:]).reshape(-1, 1)
        scaled_input = self.scaler.transform(window).reshape(1, self.timestep, 1)
        
        Xin = scaled_input.copy()
        preds = []
        for _ in range(days):
            out = self.model.predict(Xin, verbose=0)
            preds.append(float(out[0, 0]))
            Xin = self._update_input_window(Xin, out.reshape(1, 1, 1))

        preds_array = np.array(preds).reshape(-1, 1)
        final_preds = self.scaler.inverse_transform(preds_array).flatten().tolist()
        
        return final_preds

    def forecast_from_local_data(self, days: int, csv_path: str) -> List[float]:
        """Reads gold data from CSV and calls the prediction method."""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Source CSV not found at {csv_path}")

        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        
        if 'Close' not in df.columns:
            raise ValueError("CSV must contain a 'Close' column.")

        if len(df) < self.timestep:
            raise ValueError(f"CSV must contain at least {self.timestep} rows of data.")
        
        recent_closes = df['Close'].values[-self.timestep:].tolist()
        
        return self.predict_n_days(recent_closes, days)

# --- 4. FastAPI App Setup & Dependencies ---
app = FastAPI(
    title="📈 Gold Price Forecast API",
    description="A service for predicting and forecasting gold prices using an LSTM model.",
    version="1.4.0" # Version bump for correct HTTP method usage
)

def get_predictor_service() -> GoldPredictorService:
    """FastAPI Dependency: Provides the GoldPredictorService instance."""
    if _MODEL is None or _SCALER is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="ML Model and/or Scaler are not loaded. Service is unavailable."
        )
    return GoldPredictorService(_MODEL, _SCALER, TIMESTEP)


@app.on_event("startup")
def load_artifacts():
    """Load the model and scaler once when the application starts."""
    global _MODEL, _SCALER

    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print(f"ERROR: Model ({MODEL_PATH}) or scaler ({SCALER_PATH}) not found.")
        raise RuntimeError("Artifacts not found. Check paths and ensure training script was run.")
        
    try:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        _MODEL = tf.keras.models.load_model(MODEL_PATH)
        _SCALER = joblib.load(SCALER_PATH)
        print("✅ Model and Scaler loaded successfully.")
    except Exception as e:
        raise RuntimeError(f"❌ Error loading artifacts: {e}")


# --- 5. API Endpoints (Routing) ---

@app.get("/", tags=["Root"])
def home():
    """Root endpoint providing basic info."""
    return {"message": "Gold Price Prediction API is running!", "version": "1.4.0"}

@app.get("/health", tags=["Health"])
def health():
    """Standard health check endpoint."""
    return {"status": "ok", "model_loaded": _MODEL is not None}

# 👇 FIX 1: Reverted to @app.post to properly accept the JSON body data
@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(
    req: PredictRequest, # Reads data from the JSON request body
    service: GoldPredictorService = Depends(get_predictor_service)
):
    """
    Predict the next N days of gold price based on a user-provided historical time series.
    Requires a JSON body with 'recent_closes' (min 60 values).
    """
    # Pydantic validation handles the length check and required fields
    try:
        predictions = service.predict_n_days(req.recent_closes, req.days)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Prediction failed due to internal error: {e}")
        
    return {"results": predictions}

# 👇 FIX 2: Reverted to @app.post, which is appropriate for a forecast operation
@app.post("/forecast", response_model=PredictionResponse, tags=["Prediction"])
def forecast(
    req: ForecastRequest, 
    service: GoldPredictorService = Depends(get_predictor_service)
):
    """
    Forecast the next N days of gold price using the most recent data available in the local CSV file.
    Requires a JSON body with 'days' to forecast.
    """
    try:
        forecast_results = service.forecast_from_local_data(req.days, CSV_PATH)
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Forecasting failed due to internal error: {e}")
        
    return {"results": forecast_results}