from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from preprocessing import CoffeePreprocessor

app = Flask(__name__)

# Constants
VARIETIES = ['Arabica', 'Robusta']
TARGETS = ['Price', 'Demand']
SEQUENCE_LENGTH = 12

def load_all_models():
    models = {}
    for variety in VARIETIES:
        for target in TARGETS:
            model_path = f"model_{variety}_{target}.h5"
            if os.path.exists(model_path):
                models[f"{variety}_{target}"] = tf.keras.models.load_model(model_path, compile=False)
            else:
                print(f"Warning: {model_path} not found.")
    return models

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict_2026')
def predict_2026():
    try:
        data_path = 'sri_lanka_coffee_data.csv'
        if not os.path.exists(data_path):
            import Data
            Data.generate_synthetic_data()
        df = pd.read_csv(data_path)
        preprocessor = CoffeePreprocessor(sequence_length=SEQUENCE_LENGTH)
        models = load_all_models()
        
        predictions_2026 = {
            'months': ["January", "February", "March", "April", "May", "June", 
                       "July", "August", "September", "October", "November", "December"],
            'data': {}
        }

        for variety in VARIETIES:
            for target in TARGETS:
                X, y, target_col = preprocessor.prepare_data(df, variety=variety, target=target)
                model = models.get(f"{variety}_{target}")
                
                if model is None:
                    continue
                
                # Start with the last 12 months (2025 data)
                current_sequence = X[-1].reshape(1, SEQUENCE_LENGTH, 1)
                
                monthly_preds = []
                temp_seq = current_sequence.copy()
                
                # Iteratively predict 12 months of 2026
                for i in range(12):
                    pred = model.predict(temp_seq, verbose=0)
                    monthly_preds.append(pred[0, 0])
                    
                    # Update sequence: remove first, add new prediction at end
                    new_val = pred.reshape(1, 1, 1)
                    temp_seq = np.append(temp_seq[:, 1:, :], new_val, axis=1)
                
                # Inverse transform all predictions at once
                actual_preds = preprocessor.inverse_transform(np.array(monthly_preds).reshape(-1, 1), target_col)
                predictions_2026['data'][f"{variety}_{target}"] = actual_preds.flatten().tolist()

        return jsonify(predictions_2026)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
