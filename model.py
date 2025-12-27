import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Conv1D, MaxPooling1D, Flatten, Dropout

def build_hybrid_model(input_shape):
    """
    Builds a Hybrid CNN-BLSTM model.
    """
    model = Sequential()
    
    # --- CNN Layers (Feature Extraction) ---
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    
    # --- BLSTM Layers (Temporal Dependencies) ---
    model.add(Bidirectional(LSTM(50, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(50, return_sequences=False)))
    model.add(Dropout(0.2))
    
    # --- Output Layer ---
    model.add(Dense(25, activation='relu'))
    model.add(Dense(1)) # Regression output
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model
