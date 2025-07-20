"""
Data preprocessing script for emotion recognition
Loads audio files, extracts labels, and prepares data for feature extraction
"""

import os
import numpy as np
import librosa

def load_audio_data(data_dir, sr=22050):
    """
    Loads audio files and their corresponding labels from a directory structure.
    Assumes data_dir contains subdirectories named after emotions (e.g., happy, angry, sad).
    
    Args:
        data_dir (str): Path to the directory containing emotion subdirectories.
        sr (int): Target sample rate for audio loading.
        
    Returns:
        tuple: A tuple containing:
            - list: List of loaded audio signals (numpy arrays).
            - list: List of corresponding emotion labels (strings).
            - list: List of original file paths.
    """
    audio_signals = []
    labels = []
    file_paths = []
    
    for emotion_label in os.listdir(data_dir):
        emotion_dir = os.path.join(data_dir, emotion_label)
        if os.path.isdir(emotion_dir):
            for filename in os.listdir(emotion_dir):
                if filename.endswith(".wav"):
                    filepath = os.path.join(emotion_dir, filename)
                    try:
                        # Load audio with specified sample rate
                        audio, _ = librosa.load(filepath, sr=sr)
                        audio_signals.append(audio)
                        labels.append(emotion_label)
                        file_paths.append(filepath)
                    except Exception as e:
                        print(f"Error loading {filepath}: {e}")
    return audio_signals, labels, file_paths


if __name__ == "__main__":
    # Example usage:
    data_directory = "../audio_samples"
    print(f"Loading data from: {data_directory}")
    audio_data, audio_labels, audio_paths = load_audio_data(data_directory)
    
    print(f"Loaded {len(audio_data)} audio files.")
    print("First 5 labels:", audio_labels[:5])
    print("First 5 file paths:", audio_paths[:5])
    
    # You can save this processed data if needed for later steps
    # For now, we'll just demonstrate loading.
    # np.save("../data/processed_audio_data.npy", audio_data)
    # np.save("../data/processed_audio_labels.npy", audio_labels)
    print("Data preprocessing setup complete.")

