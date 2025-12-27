import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class CoffeePreprocessor:
    def __init__(self, sequence_length=12):
        """
        Args:
            sequence_length (int): Number of past months to use for prediction.
        """
        self.sequence_length = sequence_length
        self.scalers = {} # To store scalers for each Feature/Variety combination

    def create_sequences(self, data, target_col):
        """
        Creates sequences for CNN-BLSTM input.
        X shape: (samples, time_steps, features)
        y shape: (samples, 1)
        """
        X, y = [], []
        # We assume 'data' is a dataframe with relevant features. 
        # For this specific model, we might just use univariate sequences for simplicity
        # or multivariate if we include lag features. 
        # Let's stick to simple univariate prediction first: Past Price -> Next Price
        
        values = data[target_col].values
        
        # Scaling
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_values = scaler.fit_transform(values.reshape(-1, 1))
        self.scalers[target_col] = scaler
        
        for i in range(len(scaled_values) - self.sequence_length):
            X.append(scaled_values[i:i+self.sequence_length])
            y.append(scaled_values[i+self.sequence_length])
            
        return np.array(X), np.array(y)

    def inverse_transform(self, scaled_data, col_name):
        if col_name not in self.scalers:
            raise ValueError(f"Scaler for {col_name} not found.")
        return self.scalers[col_name].inverse_transform(scaled_data)

    def prepare_data(self, df, variety='Arabica', target='Price'):
        """
        Prepares X and y for a specific variety and target (Price/Demand).
        """
        target_col = f"{variety}_{target}"
        if target == 'Price':
             target_col += '_USD_kg'
        elif target == 'Demand':
             target_col += '_MT'
             
        X, y = self.create_sequences(df, target_col)
        return X, y, target_col
