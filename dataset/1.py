import os
import csv
import gc
import librosa
import numpy as np

# ================= CONFIG =================
FOLDER = "D:/UserData/Downloads/archive/Voice Emotion Dataset/fear"
OUTPUT_CSV = "fear.csv"
ERROR_LOG = "error_log.txt"

# ================= FEATURE EXTRACTION =================
def extract_features(file_path, sr=22050):
    try:
        # 🔥 Giới hạn 5s để tránh nặng RAM
        y, sr = librosa.load(file_path, sr=sr, duration=5.0)

        # ===== F0 (PITCH) =====
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7')
        )

        f0_clean = f0[~np.isnan(f0)]

        if len(f0_clean) > 0:
            F0_mean = np.mean(f0_clean)
            F0_std = np.std(f0_clean)
            F0_range = np.max(f0_clean) - np.min(f0_clean)
            Voiced_ratio = len(f0_clean) / len(f0)
        else:
            F0_mean = 0
            F0_std = 0
            F0_range = 0
            Voiced_ratio = 0

        # ===== ENERGY (RMS) =====
        energy = librosa.feature.rms(y=y)[0]
        Energy_mean = np.mean(energy)
        Energy_std = np.std(energy)

        # ===== ZCR =====
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        ZCR_mean = np.mean(zcr)
        ZCR_std = np.std(zcr)

        # ===== SPECTRAL CENTROID =====
        spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        Spectral_centroid_mean = np.mean(spec_centroid)
        Spectral_centroid_std = np.std(spec_centroid)

        # ===== STFT (GIẢM RAM) =====
        S = np.abs(librosa.stft(y, n_fft=1024))

        # ===== SPECTRAL FLUX =====
        S_db = librosa.amplitude_to_db(S)
        flux = np.sqrt(np.sum(np.diff(S_db, axis=1)**2, axis=0)) / S_db.shape[0]
        Spectral_flux_mean = np.mean(flux)

        # ===== MFCC =====
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)

        # ===== DELTA MFCC =====
        delta = librosa.feature.delta(mfcc)
        delta_mean = np.mean(delta, axis=1)
        delta_std = np.std(delta, axis=1)

        

        # ===== OUTPUT =====
        features = {
            "file": os.path.basename(file_path),

            "Energy_mean": Energy_mean,
            "Energy_std": Energy_std,

            "ZCR_mean": ZCR_mean,
            "ZCR_std": ZCR_std,

            "Spectral_centroid_mean": Spectral_centroid_mean,
            "Spectral_centroid_std": Spectral_centroid_std,

            "Spectral_flux_mean": Spectral_flux_mean,

            "F0_mean": F0_mean,
            "F0_std": F0_std,
            "F0_range": F0_range,
            "Voiced_ratio": Voiced_ratio,
        }

        # ===== MFCC =====
        for i in range(13):
            features[f"MFCC_C{i}_mean"] = mfcc_mean[i]
            features[f"MFCC_C{i}_std"] = mfcc_std[i]

        # ===== DELTA MFCC =====
        for i in range(6):
            features[f"Delta_MFCC_C{i}_mean"] = delta_mean[i]
            features[f"Delta_MFCC_C{i}_std"] = delta_std[i]

        # 🔥 Giải phóng RAM
        del y, S, S_db, mfcc, delta, f0, f0_clean
        gc.collect()

        return features

    except Exception as e:
        return str(e)


# ================= MAIN PIPELINE =================
def process_folder(folder_path, output_csv, error_log):
    if not os.path.exists(folder_path):
        print("❌ Folder không tồn tại:", folder_path)
        return

    # 🔥 Resume (không mất dữ liệu)
    processed_files = set()
    if os.path.exists(output_csv):
        with open(output_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                processed_files.add(row["file"])

    print(f"🔁 Resume: bỏ qua {len(processed_files)} file đã xử lý")

    header_written = os.path.exists(output_csv)

    files = os.listdir(folder_path)
    total = len(files)
    success = 0
    fail = 0

    for idx, file in enumerate(files):
        if not file.endswith((".wav", ".mp3", ".flac")):
            continue

        if file in processed_files:
            continue

        path = os.path.join(folder_path, file)
        print(f"[{idx+1}/{total}] Processing: {file}")

        result = extract_features(path)

        # ===== SUCCESS =====
        if isinstance(result, dict):
            with open(output_csv, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=result.keys())

                if not header_written:
                    writer.writeheader()
                    header_written = True

                writer.writerow(result)

            success += 1

        # ===== ERROR =====
        else:
            with open(error_log, "a", encoding="utf-8") as f:
                f.write(f"{file} | {result}\n")

            fail += 1

        print(f"✅ Success: {success} | ❌ Fail: {fail}")

    print("\n🎯 DONE")
    print(f"✔ Thành công: {success}")
    print(f"❌ Lỗi: {fail}")


# ================= RUN =================
if __name__ == "__main__":
    process_folder(FOLDER, OUTPUT_CSV, ERROR_LOG)