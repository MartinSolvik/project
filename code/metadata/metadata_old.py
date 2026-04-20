import os
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from scipy.signal import welch

# -------------------------
# SETTINGS
# -------------------------
DATASET_FOLDER = r"C:/Users/marti/OneDrive - NTNU/NTNU/Femte klasse/Fordypningsprosjekt/Kode/S3bucket_lovedata_2020_node7_85"
N_FILES = 303        # number of files to process
OUTPUT_FILE = "dataset_metadata.csv"

# -------------------------
# AUDIO METADATA FUNCTION
# -------------------------
def analyze_audio(file_path):

    # Load audio
    data, samplerate = sf.read(file_path)

    # Convert stereo → mono
    if len(data.shape) > 1:
        channels = data.shape[1]
        data = np.mean(data, axis=1)
    else:
        channels = 1

    n_samples = len(data)
    duration = n_samples / samplerate

    # Signal statistics
    max_val = np.max(data)
    min_val = np.min(data)
    rms = np.sqrt(np.mean(data**2))

    peak_db = 20 * np.log10(abs(max_val) + 1e-12)

    # Lmax approximation
    Lmax = 20 * np.log10(np.max(np.abs(data)) + 1e-12)

    # Spectrum
    freqs, psd = welch(data, samplerate, nperseg=4096)

    avg_spectrum = np.mean(psd)

    dominant_freq = freqs[np.argmax(psd)]

    spectral_centroid = librosa.feature.spectral_centroid(
        y=data,
        sr=samplerate
    ).mean()

    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=data,
        sr=samplerate
    ).mean()

    return {
        "filename": os.path.basename(file_path),
        "duration_sec": duration,
        "sampling_rate": samplerate,
        "samples": n_samples,
        "channels": channels,
        "max_value": max_val,
        "min_value": min_val,
        "rms": rms,
        "peak_db": peak_db,
        "Lmax": Lmax,
        "avg_spectrum_power": avg_spectrum,
        "dominant_frequency": dominant_freq,
        "spectral_centroid": spectral_centroid,
        "spectral_bandwidth": spectral_bandwidth
    }


# -------------------------
# MAIN LOOP
# -------------------------
def process_dataset(folder, n_files):

    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith((".wav", ".flac", ".mp3"))
    ]

    files = files[:n_files]

    results = []

    for f in files:
        print("Processing:", f)
        meta = analyze_audio(f)
        results.append(meta)

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False)

    print("Saved metadata to:", OUTPUT_FILE)


if _name_ == "_main_":
    process_dataset(DATASET_FOLDER, N_FILES)