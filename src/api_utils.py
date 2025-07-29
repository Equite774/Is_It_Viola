import os
from pydub import AudioSegment
import librosa
import ffmpeg
import numpy as np
from fastapi import UploadFile, File
from io import BytesIO

SAMPLE_RATE = 44100
SECONDS = 6
OFFSET = 2

def load_audio(file):
    """
    Load an audio file and return the samples.
    Parameters:
    file (BytesIO): Audio file as a BytesIO object.
    Returns:
    np.ndarray: Audio samples.
    """
    samples, _ = librosa.load(file, sr=SAMPLE_RATE)
    return samples

def split_audio(file):
    """
    Split the audio file into chunks of SECONDS length, offset by OFFSET seconds.
    Parameters:
    file (BytesIO): Audio file as a BytesIO object.
    """
    audio_data = load_audio(file)
    num_samples = len(audio_data)
    leftover = num_samples % SAMPLE_RATE
    ls = []
    for i in range(0, (num_samples - leftover) - SECONDS * SAMPLE_RATE, OFFSET * SAMPLE_RATE):
        if i + OFFSET * SAMPLE_RATE < num_samples:
            chunk = audio_data[i:i + SECONDS * SAMPLE_RATE]
            ls.append(spectogram(chunk))
    return ls

def spectogram(audio_data):
    """
    Generate a mel spectrogram from the audio data and save it as a .npy file.
    Parameters:
    audio_data (np.ndarray): Audio samples.
    start (int): Start time in samples for the chunk.
    Returns:
    List: List of mel spectrograms.
    """
    mel_sgram = librosa.power_to_db(
        librosa.feature.melspectrogram(y=audio_data, sr=SAMPLE_RATE),
        ref=np.min
    )
    mel_sgram = mel_sgram.astype(np.float32)
    return mel_sgram

def preprocess(file):
    """
    Preprocess the audio file by converting it to WAV format if necessary,
    and then splitting it into spectrogram chunks.
    Parameters:
    file (BytesIO): Audio file as a BytesIO object.
    """
    # Save the spectrograms from the audio file
    spectogram_list = split_audio(file)
    return spectogram_list