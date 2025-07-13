import numpy as np
import pandas as pd
import pydub
import librosa
from IPython.display import Audio
import sklearn as sk
import matplotlib.pyplot as plt
import torch
import torchaudio

SAMPLE_RATE = 44100
SECONDS = 2

df = pd.read_csv('metadata.csv')
data = pd.DataFrame(columns=['song', 'chunk', 'start', 'instrument', 'set'])

def load_audio(file_path):
    samples, _ = librosa.load(file_path, sr=SAMPLE_RATE)
    return samples

def add_chunk(data, start, song):
    data.append({
        'song': song['filename'],
        'start': start,
        'instrument': song['instrument'],
        'set': song['set']
    })

def data_from_file(data, song):
    audio_data = load_audio(song['filename'])
    num_samples = len(audio_data)
    leftover = num_samples % SAMPLE_RATE
    for i in range(0, (num_samples - leftover) - 10 * SAMPLE_RATE, SECONDS * SAMPLE_RATE):
        if i + SECONDS * SAMPLE_RATE < num_samples:
            add_chunk(data, i, song)

def parse_metadata(df, data):
    for i, song in df.iterrows():
        if i % 25 == 0:
            print(f"Processing {song['filename']} with instrument {song['instrument']} and set {song['set']}")
        data_from_file(data, song)

def save_data(data):
    processed_data = pd.DataFrame(data)
    processed_data.to_csv('processed_data.csv', index=False)

def main():
    df = pd.read_csv('metadata.csv')
    data = []
    parse_metadata(df, data)
    save_data(data)

if __name__ == "__main__":
    main()