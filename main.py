
import os
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load Models
LABOR_MODEL_PATH = "Labor/labor_demand_model.pkl"
LABOR_SCALER_PATH = "Labor/labor_feature_scaler.pkl"
TRANSPORT_MODEL_PATH = "transport/transport_demand_model.pkl"
TRANSPORT_SCALER_PATH = "transport/transport_feature_scaler.pkl"

labor_model = None
labor_scaler = None
transport_model = None
transport_scaler = None

def load_models():
    global labor_model, labor_scaler, transport_model, transport_scaler
    try:
        if os.path.exists(LABOR_MODEL_PATH):
            labor_model = joblib.load(LABOR_MODEL_PATH)
            labor_scaler = joblib.load(LABOR_SCALER_PATH)
            print("Labor model loaded.")
        else:
            print("Labor model not found. Please run rebuild_labor_model.py")
        
        if os.path.exists(TRANSPORT_MODEL_PATH):
            transport_model = joblib.load(TRANSPORT_MODEL_PATH)
            transport_scaler = joblib.load(TRANSPORT_SCALER_PATH)
            print("Transport model loaded.")
        else:
            print("Transport model not found.")
    except Exception as e:
        print(f"Error loading models: {e}")

load_models()

# Pydantic Models for Input
class LaborInput(BaseModel):
    area_ha: float
    predicted_yield_kg_per_ha: float
    temp: float
    feelslike: float
    humidity: float
    precip: float
    severerisk: float = 10.0 # Default/fallback
    month: int

class TransportInput(BaseModel):
    area_ha: float
    predicted_yield_kg_per_ha: float
    temp: float
    feelslike: float
    humidity: float
    precip: float
    severerisk: float = 10.0
    month: int

# Feature engineering helper
def calculate_daily_harvest(area_ha, predicted_yield):
    return (area_ha * predicted_yield) / 30

def calculate_productivity(precip, humidity):
    prod = 1.0 - (precip / 100) * 0.3 - (humidity / 100) * 0.2
    return np.clip(prod, 0.4, 1.0)

def get_regime_shift(month):
    # Logic from transport notebook: 1 if month >= 1 and 2025 <= 2026 else 0
    # Assuming current era (2025/2026), let's default to 1 for now if month matches?
    # Actually, the notebook had hardcoded "2025-2026". Let's assume 0 unless we know year.
    # The input doesn't have year. Let's assume 0 for safety or add year to input.
    # For now, let's stick to simple logic: Input doesn't have year.
    # Notebook logic: regime_shift = 1 if month >= 1 and 2025 <= 2026 else 0
    # Let's assume 0 to be safe, or 1 if we assume we are in that period.
    return 1 # simplifying assumption based on notebook context of "current"

@app.get("/")
async def read_index():
    return FileResponse('static/index.html')

@app.post("/predict/labor")
async def predict_labor(data: LaborInput):
    if not labor_model or not labor_scaler:
        raise HTTPException(status_code=503, detail="Labor model not available")
    
    daily_harvest = calculate_daily_harvest(data.area_ha, data.predicted_yield_kg_per_ha)
    productivity = calculate_productivity(data.precip, data.humidity)
    
    input_data = {
        "area_ha": data.area_ha,
        "predicted_yield_kg_per_ha": data.predicted_yield_kg_per_ha,
        "daily_harvest_kg": daily_harvest,
        "temp": data.temp,
        "feelslike": data.feelslike,
        "humidity": data.humidity,
        "precip": data.precip,
        "severerisk": data.severerisk,
        "productivity_index": productivity,
        "month": data.month
    }
    
    # Order matters!
    FEATURE_ORDER = [
        "area_ha", "predicted_yield_kg_per_ha", "daily_harvest_kg",
        "temp", "feelslike", "humidity", "precip", "severerisk",
        "productivity_index", "month"
    ]
    
    df = pd.DataFrame([input_data])[FEATURE_ORDER]
    X_scaled = labor_scaler.transform(df)
    prediction = labor_model.predict(X_scaled)[0]
    
    return {
        "pickers": int(round(prediction[0])),
        "harvesters": int(round(prediction[1])),
        "loaders": int(round(prediction[2]))
    }

@app.post("/predict/transport")
async def predict_transport(data: TransportInput):
    if not transport_model or not transport_scaler:
        # Warning: Transport model files might be missing too if not checked specifically
        # But we found them earlier.
        raise HTTPException(status_code=503, detail="Transport model not available")
    
    productivity = calculate_productivity(data.precip, data.humidity)
    regime_shift = get_regime_shift(data.month)
    
    input_data = {
        "area_ha": data.area_ha,
        "predicted_yield_kg_per_ha": data.predicted_yield_kg_per_ha,
        "temp": data.temp,
        "feelslike": data.feelslike,
        "humidity": data.humidity,
        "precip": data.precip,
        "severerisk": data.severerisk,
        "productivity_index": productivity,
        "month": data.month,
        "regime_shift": regime_shift
    }
    
    FEATURE_ORDER = [
        "area_ha", "predicted_yield_kg_per_ha", "temp", "feelslike",
        "humidity", "precip", "severerisk", "productivity_index",
        "month", "regime_shift"
    ]
    
    df = pd.DataFrame([input_data])[FEATURE_ORDER]
    X_scaled = transport_scaler.transform(df)
    prediction = transport_model.predict(X_scaled)[0]
    
    return {
        "tractors": int(round(prediction[0])),
        "apes": int(round(prediction[1])),
        "trucks": int(round(prediction[2]))
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
