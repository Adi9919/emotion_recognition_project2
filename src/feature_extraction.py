"""
Feature extraction script for emotion recognition
Extracts MFCCs from audio signals and prepares them for model input
"""

import librosa
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

# Assuming data_preprocessing.py is in the same directory
from data_preprocessing import load_audio_data

def extract_mfccs(audio, sr, n_mfcc=40, hop_length=512, n_fft=2048):
    """
    Extracts Mel-Frequency Cepstral Coefficients (MFCCs) from an audio signal.
    
    Args:
        audio (np.array): Audio time series.
        sr (int): Sampling rate of the audio.
        n_mfcc (int): Number of MFCCs to return.
        hop_length (int): The number of samples between successive frames.
        n_fft (int): Length of the FFT window.
        
    Returns:
        np.array: Extracted MFCCs, typically of shape (n_mfcc, number_of_frames).
    """
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)
    return mfccs

def pad_mfccs(mfccs, max_padding_length=100):
    """
    Pads or truncates MFCC sequences to a fixed length.
    
    Args:
        mfccs (np.array): MFCC array of shape (n_mfcc, number_of_frames).
        max_padding_length (int): The target length for the number of frames.
        
    Returns:
        np.array: Padded or truncated MFCC array.
    """
    if mfccs.shape[1] > max_padding_length:
        # Truncate
        mfccs = mfccs[:, :max_padding_length]
    elif mfccs.shape[1] < max_padding_length:
        # Pad with zeros
        pad_width = max_padding_length - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return mfccs

def preprocess_features(features):
    """
    Standardizes the features using StandardScaler.
    
    Args:
        features (np.array): Features array, e.g., (num_samples, n_mfcc, padded_length).
        
    Returns:
        np.array: Standardized features.
        StandardScaler: Fitted scaler object.
    """
    # Reshape for StandardScaler: (num_samples * n_mfcc * padded_length, 1)
    original_shape = features.shape
    features_reshaped = features.reshape(-1, features.shape[-1]) # Flatten last dimension for scaling
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_reshaped)
    
    # Reshape back to original dimensions
    scaled_features = scaled_features.reshape(original_shape)
    
    return scaled_features, scaler


if __name__ == "__main__":
    data_directory = "../audio_samples"
    audio_signals, labels, _ = load_audio_data(data_directory)
    
    all_mfccs = []
    for audio in audio_signals:
        mfccs = extract_mfccs(audio, sr=22050)
        all_mfccs.append(mfccs)
    
    # Determine maximum length for padding
    max_len = max([mfcc.shape[1] for mfcc in all_mfccs])
    print(f"Maximum MFCC sequence length: {max_len}")
    
    # Pad all MFCCs to the maximum length found
    padded_mfccs = [pad_mfccs(mfcc, max_padding_length=max_len) for mfcc in all_mfccs]
    
    # Convert list of arrays to a single numpy array
    X = np.array(padded_mfccs)
    
    # Reshape for CNN input: (num_samples, n_mfcc, padded_length, 1)
    X = X[..., np.newaxis]
    
    print(f"Shape of MFCC features (X): {X.shape}")
    
    # Example of feature scaling (optional, but good practice)
    # X_scaled, scaler = preprocess_features(X)
    # print(f"Shape of scaled MFCC features (X_scaled): {X_scaled.shape}")
    
    # Save features and labels for later use
    np.save("../data/mfcc_features.npy", X)
    np.save("../data/labels.npy", labels)
    print("MFCC features and labels saved to ../data/")

    print("Feature extraction setup complete.")

