"""
Audio sample generator for emotion recognition testing
Creates synthetic audio samples with different characteristics for happy, angry, and sad emotions
"""

import numpy as np
import librosa
import soundfile as sf
import os

def generate_synthetic_audio(emotion, duration=3.0, sr=22050):
    """
    Generate synthetic audio samples with characteristics typical of different emotions
    
    Args:
        emotion (str): 'happy', 'angry', or 'sad'
        duration (float): Duration in seconds
        sr (int): Sample rate
    
    Returns:
        np.array: Audio signal
    """
    t = np.linspace(0, duration, int(sr * duration))
    
    if emotion == 'happy':
        # Higher pitch, more variation, upward intonation
        base_freq = 200
        freq_variation = 50 * np.sin(2 * np.pi * 2 * t)  # 2 Hz variation
        pitch_trend = 20 * t  # Upward trend
        frequency = base_freq + freq_variation + pitch_trend
        
        # Add harmonics for brightness
        signal = (np.sin(2 * np.pi * frequency * t) + 
                 0.3 * np.sin(2 * np.pi * 2 * frequency * t) +
                 0.1 * np.sin(2 * np.pi * 3 * frequency * t))
        
        # Add some jitter for liveliness
        signal += 0.1 * np.random.normal(0, 0.1, len(signal))
        
    elif emotion == 'angry':
        # Lower pitch, harsh, more noise
        base_freq = 120
        freq_variation = 30 * np.sin(2 * np.pi * 3 * t)  # 3 Hz variation, more aggressive
        frequency = base_freq + freq_variation
        
        # Harsh harmonics
        signal = (np.sin(2 * np.pi * frequency * t) + 
                 0.5 * np.sin(2 * np.pi * 2 * frequency * t) +
                 0.3 * np.sin(2 * np.pi * 3 * frequency * t))
        
        # Add noise for harshness
        signal += 0.2 * np.random.normal(0, 0.2, len(signal))
        
        # Add some distortion
        signal = np.tanh(2 * signal)
        
    elif emotion == 'sad':
        # Lower pitch, monotone, downward intonation
        base_freq = 100
        freq_variation = 10 * np.sin(2 * np.pi * 0.5 * t)  # Slow, small variation
        pitch_trend = -10 * t  # Downward trend
        frequency = base_freq + freq_variation + pitch_trend
        
        # Simple, muted tone
        signal = np.sin(2 * np.pi * frequency * t)
        
        # Add slight tremolo for sadness
        tremolo = 1 + 0.1 * np.sin(2 * np.pi * 4 * t)
        signal *= tremolo
        
        # Reduce amplitude over time (fading)
        envelope = np.exp(-0.3 * t)
        signal *= envelope
    
    # Normalize
    signal = signal / np.max(np.abs(signal)) * 0.8
    
    return signal

def create_sample_dataset(output_dir, num_samples_per_emotion=5):
    """
    Create a sample dataset with synthetic audio files
    
    Args:
        output_dir (str): Directory to save audio files
        num_samples_per_emotion (int): Number of samples per emotion
    """
    emotions = ['happy', 'angry', 'sad']
    
    for emotion in emotions:
        emotion_dir = os.path.join(output_dir, emotion)
        os.makedirs(emotion_dir, exist_ok=True)
        
        for i in range(num_samples_per_emotion):
            # Vary duration slightly
            duration = 2.5 + np.random.uniform(-0.5, 1.0)
            audio = generate_synthetic_audio(emotion, duration)
            
            filename = f"{emotion}_{i+1:02d}.wav"
            filepath = os.path.join(emotion_dir, filename)
            
            sf.write(filepath, audio, 22050)
            print(f"Created: {filepath}")

if __name__ == "__main__":
    # Create sample dataset
    output_dir = "../audio_samples"
    create_sample_dataset(output_dir)
    print("Sample dataset created successfully!")

