"""
Prediction script for emotion recognition
Loads a trained model and predicts emotion from a given audio file.
"""

import tensorflow as tf
from tensorflow.keras.models import load_model
import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

# Assuming feature_extraction.py is in the same directory
from feature_extraction import extract_mfccs, pad_mfccs

def predict_emotion(audio_path, model_path, label_encoder_classes_path, sr=22050, max_padding_length=151):
    """
    Predicts the emotion from an audio file using a trained model.
    
    Args:
        audio_path (str): Path to the audio file.
        model_path (str): Path to the trained model (.h5 file).
        label_encoder_classes_path (str): Path to the saved label encoder classes (.npy file).
        sr (int): Sample rate for audio loading.
        max_padding_length (int): Max length for MFCC padding (should match training).
        
    Returns:
        str: Predicted emotion label.
    """
    try:
        # Load audio
        audio, _ = librosa.load(audio_path, sr=sr)
        
        # Extract MFCCs
        mfccs = extract_mfccs(audio, sr=sr)
        
        # Pad MFCCs
        padded_mfccs = pad_mfccs(mfccs, max_padding_length=max_padding_length)
        
        # Reshape for model input (add batch dimension and channel dimension)
        # For CNN: (1, n_mfcc, padded_length, 1)
        # For RNN/LSTM: (1, padded_length, n_mfcc)
        if "cnn" in model_path:
            X_input = padded_mfccs[np.newaxis, ..., np.newaxis]
        else: # RNN/LSTM
            X_input = padded_mfccs.transpose(1, 0)[np.newaxis, ...]

        # Load model
        model = load_model(model_path)
        
        # Load label encoder classes
        le_classes = np.load(label_encoder_classes_path, allow_pickle=True)
        le = LabelEncoder()
        le.fit(le_classes) # Fit with the loaded classes

        # Predict
        predictions = model.predict(X_input)
        predicted_class_index = np.argmax(predictions)
        predicted_emotion = le.inverse_transform([predicted_class_index])[0]
        
        return predicted_emotion
        
    except Exception as e:
        return f"Error during prediction: {e}"

if __name__ == "__main__":
    # Example usage:
    audio_sample_dir = "../audio_samples"
    cnn_model_path = "../models/cnn_emotion_model.h5"
    rnn_lstm_model_path = "../models/rnn_lstm_emotion_model.h5"
    label_encoder_classes_path = "../models/label_encoder_classes.npy"

    print("\n--- Testing CNN Model ---")
    for emotion_folder in os.listdir(audio_sample_dir):
        emotion_path = os.path.join(audio_sample_dir, emotion_folder)
        if os.path.isdir(emotion_path):
            for audio_file in os.listdir(emotion_path):
                if audio_file.endswith(".wav"):
                    full_audio_path = os.path.join(emotion_path, audio_file)
                    predicted = predict_emotion(full_audio_path, cnn_model_path, label_encoder_classes_path)
                    print(f"Audio: {audio_file}, Predicted Emotion (CNN): {predicted}")

    print("\n--- Testing RNN/LSTM Model ---")
    for emotion_folder in os.listdir(audio_sample_dir):
        emotion_path = os.path.join(audio_sample_dir, emotion_folder)
        if os.path.isdir(emotion_path):
            for audio_file in os.listdir(emotion_path):
                if audio_file.endswith(".wav"):
                    full_audio_path = os.path.join(emotion_path, audio_file)
                    predicted = predict_emotion(full_audio_path, rnn_lstm_model_path, label_encoder_classes_path)
                    print(f"Audio: {audio_file}, Predicted Emotion (RNN/LSTM): {predicted}")

    print("Prediction demonstration complete.")

