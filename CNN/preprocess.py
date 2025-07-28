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
SECONDS = 6
OFFSET = 2

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
    for i in range(0, (num_samples - leftover) - SECONDS * SAMPLE_RATE, OFFSET * SAMPLE_RATE):
        if i + OFFSET * SAMPLE_RATE < num_samples:
            add_chunk(data, i, song)

def parse_metadata(df, data):
    for i, song in df.iterrows():
        if i % 25 == 0:
            print(f"Processing {song['filename']} with instrument {song['instrument']} and set {song['set']}")
        data_from_file(data, song)

def save_data(data):
    processed_data = pd.DataFrame(data)
    processed_data.to_csv('CNN/processed_chunks.csv', index=False)

def spectogram():
    df = pd.read_csv('CNN/processed_chunks.csv')
    for i, row in df.iterrows():
        audio_data = load_audio(row['song'])
        start = row['start']
        directory = 'CNN/train' if row['set'] == 'train' else 'CNN/test'
        chunk = audio_data[start:start + SECONDS * SAMPLE_RATE]
        mel_sgram = librosa.power_to_db(
            librosa.feature.melspectrogram(y=chunk, sr=SAMPLE_RATE),
            ref=np.min
        )
        np.save(f"{directory}/mel_{row['song'].split('/')[-1].replace('.wav', '')}_start_{row['start'] / SAMPLE_RATE}.npy", mel_sgram)

def main():
    df = pd.read_csv('data/metadata.csv')
    data = []
    parse_metadata(df, data)
    save_data(data)
    spectogram()

if __name__ == "__main__":
    main()