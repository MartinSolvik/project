import os
import csv
import argparse
import tempfile
import numpy as np
import soundfile as sf
import librosa
import boto3
from scipy.signal import welch

# Settings

AWS_ACCESS_KEY = "SIhhsZjApucid3WetEFe"
AWS_SECRET_KEY = "/L868ZMXQMHBvGyLUlf955+MchKsSVvLsC3rkwQ2"

S3_BUCKET = "lovedata"
OUTPUT_FILE = "dataset_metadata.csv"
PROGRESS_FILE = "processed_files.txt"
AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3")

FIELDNAMES = [
    "filename", "s3_key", "duration_sec", "sampling_rate", "bit_depth", "samples",
    "channels", "max_value", "min_value", "rms", "peak_db", "Lmax",
    "avg_spectrum_power", "dominant_frequency", "spectral_centroid", "spectral_bandwidth"
]


# S3 client

def get_s3():
    return boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        endpoint_url="https://s3.hi.no",
    )


# Progress tracking

def load_processed(progress_file):
    if not os.path.exists(progress_file):
        return set()
    with open(progress_file, "r") as f:
        return set(line.strip() for line in f if line.strip())

def mark_processed(progress_file, filename):
    with open(progress_file, "a") as f:
        f.write(filename + "\n")


# S3 helpers

def list_folders(bucket):
    s3 = get_s3()
    paginator = s3.get_paginator("list_objects_v2")
    folders = set()
    for page in paginator.paginate(Bucket=bucket, Delimiter="/"):
        for prefix in page.get("CommonPrefixes", []):
            folders.add(prefix["Prefix"].rstrip("/"))
    return sorted(folders)

def list_s3_files(bucket, folder):
    prefix = folder.rstrip("/") + "/" if folder else ""
    s3 = get_s3()
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.lower().endswith(AUDIO_EXTENSIONS):
                keys.append(key)
    return keys

def download_file(bucket, key, local_dir):
    s3 = get_s3()
    filename = os.path.basename(key)
    local_path = os.path.join(local_dir, filename)
    s3.download_file(bucket, key, local_path)
    return local_path


# Audio analysis

def analyze_audio(file_path):
    info = sf.info(file_path)
    data, samplerate = sf.read(file_path, always_2d=False)

    # bit depth from subtype, e.g. PCM_16 -> 16
    subtype = info.subtype
    if "8" in subtype:
        bit_depth = 8
    elif "16" in subtype:
        bit_depth = 16
    elif "24" in subtype:
        bit_depth = 24
    elif "32" in subtype:
        bit_depth = 32
    elif "64" in subtype:
        bit_depth = 64
    else:
        bit_depth = subtype

    if len(data.shape) > 1:
        channels = data.shape[1]
        data = np.mean(data, axis=1)
    else:
        channels = 1

    n_samples = len(data)
    duration = n_samples / samplerate

    max_val = float(np.max(data))
    min_val = float(np.min(data))
    rms = float(np.sqrt(np.mean(data**2)))
    peak_db = 20 * np.log10(abs(max_val) + 1e-12)
    Lmax = 20 * np.log10(np.max(np.abs(data)) + 1e-12)

    freqs, psd = welch(data, samplerate, nperseg=4096)
    avg_spectrum = float(np.mean(psd))
    dominant_freq = float(freqs[np.argmax(psd)])

    spectral_centroid = float(librosa.feature.spectral_centroid(y=data, sr=samplerate).mean())
    spectral_bandwidth = float(librosa.feature.spectral_bandwidth(y=data, sr=samplerate).mean())

    return {
        "filename": os.path.basename(file_path),
        "s3_key": "",
        "duration_sec": duration,
        "sampling_rate": samplerate,
        "bit_depth": bit_depth,
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
        "spectral_bandwidth": spectral_bandwidth,
    }


# CSV helpers

def init_csv(output_file):
    if not os.path.exists(output_file):
        with open(output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()

def append_csv(output_file, row):
    with open(output_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writerow(row)


# Main

def main():
    parser = argparse.ArgumentParser(description="Analyze audio metadata from S3 bucket lovedata")
    parser.add_argument("--folder", type=str, default=None,
                        help="Folder inside the bucket to process. Omit to list available folders.")
    parser.add_argument("--n", type=int, default=None,
                        help="Max number of files to process (default: all unprocessed)")
    parser.add_argument("--output", type=str, default=OUTPUT_FILE,
                        help="Output CSV file (default: dataset_metadata.csv)")
    parser.add_argument("--progress", type=str, default=PROGRESS_FILE,
                        help="Progress tracking file (default: processed_files.txt)")
    args = parser.parse_args()

    # list folders and exit if no folder given
    if args.folder is None:
        print("No --folder specified. Available folders in s3://" + S3_BUCKET + ":\n")
        folders = list_folders(S3_BUCKET)
        if not folders:
            print("  (no top-level folders found)")
        for f in folders:
            print("  " + f)
        print("\nRe-run with: --folder <folder_name>")
        return

    print("Listing files in s3://" + S3_BUCKET + "/" + args.folder + " ...")
    all_keys = list_s3_files(S3_BUCKET, args.folder)
    print("Found " + str(len(all_keys)) + " audio files")

    processed = load_processed(args.progress)
    pending = [k for k in all_keys if os.path.basename(k) not in processed]
    print("Already processed: " + str(len(processed)) + " | Remaining: " + str(len(pending)))

    if args.n is not None:
        pending = pending[:args.n]
    print("Will process: " + str(len(pending)) + " files\n")

    if not pending:
        print("Nothing to do.")
        return

    init_csv(args.output)

    with tempfile.TemporaryDirectory() as tmpdir:
        done = 0
        errors = 0

        for i, key in enumerate(pending, 1):
            filename = os.path.basename(key)
            print("[" + str(i) + "/" + str(len(pending)) + "] " + filename)

            local_path = None
            try:
                print("  Downloading ...", end=" ", flush=True)
                local_path = download_file(S3_BUCKET, key, tmpdir)
                print("done")

                print("  Analyzing   ...", end=" ", flush=True)
                meta = analyze_audio(local_path)
                meta["s3_key"] = key
                print("done")

                print("  Saving      ...", end=" ", flush=True)
                append_csv(args.output, meta)
                mark_processed(args.progress, filename)
                print("done")

                print("  Deleting    ...", end=" ", flush=True)
                if local_path and os.path.exists(local_path):
                    os.remove(local_path)
                    local_path = None
                print("done")

                done += 1

            except Exception as e:
                errors += 1
                print("ERROR: " + str(e))

            finally:
                if local_path and os.path.exists(local_path):
                    os.remove(local_path)

    print("\nDone. Processed: " + str(done) + " | Errors: " + str(errors))
    print("Results saved to: " + args.output)
    print("Progress saved to: " + args.progress)


if __name__ == "__main__":
    main()
