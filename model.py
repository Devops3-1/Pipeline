from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

def train_model():
    # Simple model that adds two numbers
    model = Sequential([
        Dense(10, input_shape=(2,), activation='relu'),
        Dense(1)
    ])
    
    # Fake training data (just for demonstration)
    x_train = np.array([[1, 2], [3, 4], [5, 6]])
    y_train = np.sum(x_train, axis=1)

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=10, verbose=0)

    return model
