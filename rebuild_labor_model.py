
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import os

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Labor", "data", "synthetic_daily_labor_dataset_REAL_WEATHER.csv")
MODEL_PATH = os.path.join(BASE_DIR, "Labor", "labor_demand_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "Labor", "labor_feature_scaler.pkl")

print(f"Loading data from: {DATA_PATH}")

# Load Labor Dataset
df = pd.read_csv(DATA_PATH)
df["datetime"] = pd.to_datetime(df["datetime"])

# Feature / Target Definition
TARGETS = [
    "pickers_needed",
    "harvesters_needed",
    "loaders_needed"
]

FEATURES = [
    "area_ha",
    "predicted_yield_kg_per_ha",
    "daily_harvest_kg",
    "temp",
    "feelslike",
    "humidity",
    "precip",
    "severerisk",
    "productivity_index",
    "month"
]

# Ensure Month is present (extract if needed, though the CSV seems to have it)
if "month" not in df.columns:
    df["month"] = df["datetime"].dt.month

# Clean data if necessary (replace NaNs in severerisk if any, though XGBoost handles it)
# The notebook showed severerisk NaN in head(), but let's assume it's handled or we fill it.
# Based on the notebook, it seems XGBoost handles it, but let's double check if we need to fill.
# The notebook didn't explicitly fill NaNs before training, so we'll leave it to XGBoost.

X = df[FEATURES]
y = df[TARGETS]

print("Splitting data...")
# Train / Validation Split
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    shuffle=False
)

print("Scaling features...")
# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

print("Training model...")
# Build AI Model (Multi-Output XGBoost)
base_model = XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42
)

model = MultiOutputRegressor(base_model)

# Train the AI Model
model.fit(X_train_scaled, y_train)

print(f"Saving model to {MODEL_PATH}...")
joblib.dump(model, MODEL_PATH)

print(f"Saving scaler to {SCALER_PATH}...")
joblib.dump(scaler, SCALER_PATH)

print("Done.")
