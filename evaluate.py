import pandas as pd
import numpy as np
import os
from preprocessing import CoffeePreprocessor
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_models():
    data_path = 'sri_lanka_coffee_data.csv'
    df = pd.read_csv(data_path)
    preprocessor = CoffeePreprocessor(sequence_length=12)
    
    varieties = ['Arabica', 'Robusta']
    targets = ['Price', 'Demand']
    
    summary = {}
    
    for variety in varieties:
        for target in targets:
            # Load model
            model_path = f"model_{variety}_{target}.h5"
            if not os.path.exists(model_path):
                continue
            
            model = tf.keras.models.load_model(model_path, compile=False)
            
            # Prepare Data
            X, y, target_col = preprocessor.prepare_data(df, variety=variety, target=target)
            
            # Split (same as training)
            split = int(len(X) * 0.8)
            X_test, y_test = X[split:], y[split:]
            
            # Predict
            predictions = model.predict(X_test, verbose=0)
            
            # Inverse Transform
            y_test_inv = preprocessor.inverse_transform(y_test.reshape(-1, 1), target_col)
            predictions_inv = preprocessor.inverse_transform(predictions, target_col)
            
            mae = mean_absolute_error(y_test_inv, predictions_inv)
            rmse = np.sqrt(mean_squared_error(y_test_inv, predictions_inv))
            
            # For Demand, we can also calculate MAPE (Mean Absolute Percentage Error)
            mape = np.mean(np.abs((y_test_inv - predictions_inv) / y_test_inv)) * 100
            
            summary[f"{variety} {target}"] = {
                'MAE': f"{mae:.2f}",
                'RMSE': f"{rmse:.2f}",
                'MAPE': f"{mape:.2f}%"
            }
            
    return summary

if __name__ == "__main__":
    results = evaluate_models()
    print("--- Model Accuracy Summary ---")
    for key, val in results.items():
        print(f"{key}: {val}")
