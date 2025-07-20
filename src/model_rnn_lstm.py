"""
RNN/LSTM model for emotion recognition
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import numpy as np
import os

# Load features and labels
X = np.load("../data/mfcc_features.npy")
y = np.load("../data/labels.npy")

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Reshape X for RNN/LSTM input (samples, timesteps, features)
# Current shape: (num_samples, n_mfcc, padded_length, 1)
# Desired shape: (num_samples, padded_length, n_mfcc)
X_reshaped = X.squeeze(axis=-1).transpose(0, 2, 1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_categorical, test_size=0.2, random_state=42)

def create_rnn_lstm_model(input_shape, num_classes):
    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape),
        Dropout(0.3),
        Bidirectional(LSTM(64)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    input_shape = X_train.shape[1:]
    num_classes = y_categorical.shape[1]
    
    model = create_rnn_lstm_model(input_shape, num_classes)
    model.summary()
    
    # Train the model
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
    
    # Save the model
    model.save("../models/rnn_lstm_emotion_model.h5")
    print("RNN/LSTM model trained and saved successfully!")

    # Label encoder classes are already saved by CNN model, no need to save again


