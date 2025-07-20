"""
Model evaluation script for emotion recognition
Evaluates trained CNN and RNN/LSTM models
"""

import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
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
X_reshaped = X.squeeze(axis=-1).transpose(0, 2, 1)

# Split data (using the same split as training for consistency in evaluation)
# For CNN
X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = train_test_split(X, y_categorical, test_size=0.2, random_state=42)
# For RNN/LSTM
X_train_rnn, X_test_rnn, y_train_rnn, y_test_rnn = train_test_split(X_reshaped, y_categorical, test_size=0.2, random_state=42)

def evaluate_model(model_path, X_test, y_test, label_encoder_classes):
    """
    Loads a trained model and evaluates its performance.
    
    Args:
        model_path (str): Path to the saved model (.h5 file).
        X_test (np.array): Test features.
        y_test (np.array): True labels for the test set (one-hot encoded).
        label_encoder_classes (np.array): Array of original class names.
    """
    print(f"\n--- Evaluating Model: {os.path.basename(model_path)} ---")
    model = load_model(model_path)
    
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=label_encoder_classes))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    label_encoder_classes = np.load("../models/label_encoder_classes.npy", allow_pickle=True)

    # Evaluate CNN model
    cnn_model_path = "../models/cnn_emotion_model.h5"
    if os.path.exists(cnn_model_path):
        evaluate_model(cnn_model_path, X_test_cnn, y_test_cnn, label_encoder_classes)
    else:
        print(f"CNN model not found at {cnn_model_path}")

    # Evaluate RNN/LSTM model
    rnn_lstm_model_path = "../models/rnn_lstm_emotion_model.h5"
    if os.path.exists(rnn_lstm_model_path):
        evaluate_model(rnn_lstm_model_path, X_test_rnn, y_test_rnn, label_encoder_classes)
    else:
        print(f"RNN/LSTM model not found at {rnn_lstm_model_path}")

    print("Model evaluation complete.")

