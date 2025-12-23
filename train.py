import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from preprocessing import CoffeePreprocessor
from model import build_hybrid_model
from sklearn.metrics import mean_absolute_error, mean_squared_error

def train_and_evaluate():
    # Load Data
    data_path = 'sri_lanka_coffee_data.csv'
    if not os.path.exists(data_path):
        print("Data file not found. Running generator...")
        import Data
        Data.generate_synthetic_data()
        
    df = pd.read_csv(data_path)
    
    preprocessor = CoffeePreprocessor(sequence_length=12)
    
    varieties = ['Arabica', 'Robusta']
    targets = ['Price', 'Demand']
    
    results = {}
    
    for variety in varieties:
        for target in targets:
            print(f"\nTraining model for {variety} - {target}...")
            
            # Prepare Data
            X, y, target_col = preprocessor.prepare_data(df, variety=variety, target=target)
            
            # Split
            split = int(len(X) * 0.8)
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]
            
            # Reshape for CNN (samples, steps, features)
            # Current X is (samples, 12, 1) if we just reshape
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
            
            # Build Model
            model = build_hybrid_model(input_shape=(X_train.shape[1], 1))
            
            # Train
            history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test), verbose=0)
            
            # Save Model
            model_name = f"model_{variety}_{target}.h5"
            model.save(model_name)
            print(f"Model saved as {model_name}")
            
            # Evaluate
            predictions = model.predict(X_test)
            
            # Inverse Transform
            y_test_inv = preprocessor.inverse_transform(y_test.reshape(-1, 1), target_col)
            predictions_inv = preprocessor.inverse_transform(predictions, target_col)
            
            mae = mean_absolute_error(y_test_inv, predictions_inv)
            rmse = np.sqrt(mean_squared_error(y_test_inv, predictions_inv))
            
            print(f"MAE: {mae:.2f}")
            print(f"RMSE: {rmse:.2f}")
            
            results[f"{variety}_{target}"] = {'mae': mae, 'rmse': rmse}
            
            # Plot
            plt.figure(figsize=(10, 6))
            plt.plot(y_test_inv, label='Actual')
            plt.plot(predictions_inv, label='Predicted')
            plt.title(f"{variety} {target} Prediction (Test Set)")
            plt.legend()
            plt.savefig(f"plot_{variety}_{target}.png")
            plt.close()

    print("\nTraining Complete. Summary:")
    print(results)

if __name__ == "__main__":
    train_and_evaluate()
