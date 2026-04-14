"""
Tạo dataset/anger.csv từ các file âm thanh trong Voice_Emotion_Dataset/anger.
Cột và thư viện tham chiếu: dataset/README.md

Tiến trình lưu vào anger_partial.csv (có cột source_file). Tắt giữa chừng rồi chạy lại
sẽ bỏ qua file đã xong và tiếp tục. Khi xử lý hết, ghi anger.csv và xóa partial.
"""

from __future__ import annotations

import glob
import sys
import time
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np
import pandas as pd


# Cột CSV theo thứ tự README.md
CSV_COLUMNS = [
    "F0_mean",
    "F0_std",
    "F0_range",
    "Energy_ mean",
    "Energy_ std",
    "ZCR_mean",
    "ZCR_std",
    "Spectral_centroid_mean",
    "Spectral_centroid_std",
    "Spectral_flux_mean",
    *[f"MFCC_C{i}_mean" for i in range(13)],
    *[f"MFCC_C{i}_std" for i in range(13)],
    *[f"Delta_MFCC_C{i}_mean" for i in range(6)],
    *[f"Delta_MFCC_C{i}_std" for i in range(6)],
]

SOURCE_COL = "source_file"
PARTIAL_COLUMNS = [SOURCE_COL] + CSV_COLUMNS
PARTIAL_NAME = "anger_partial.csv"


def _rel_key(audio_path: Path, anger_dir: Path) -> str:
    return audio_path.resolve().relative_to(anger_dir.resolve()).as_posix()


def _spectral_flux_mean(y: np.ndarray, sr: int) -> float:
    """Trung bình spectral flux theo khung (chênh lệch phổ liên tiếp)."""
    S = np.abs(librosa.stft(y))
    if S.shape[1] < 2:
        return 0.0
    diff = np.diff(S, axis=1)
    flux_per_frame = np.sqrt(np.mean(diff**2, axis=0))
    return float(np.mean(flux_per_frame))


def extract_row(audio_path: Path, sr: int | None = None) -> dict[str, float]:
    y, sr = librosa.load(str(audio_path), sr=sr, mono=True)

    f0, _, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
    )
    f0_v = f0[~np.isnan(f0)]
    if f0_v.size == 0:
        f0_mean = f0_std = f0_range = 0.0
    else:
        f0_mean = float(np.mean(f0_v))
        f0_std = float(np.std(f0_v))
        f0_range = float(np.ptp(f0_v))

    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    delta_mfcc = librosa.feature.delta(mfcc)
    d06 = delta_mfcc[:6]

    row: dict[str, float] = {
        "F0_mean": f0_mean,
        "F0_std": f0_std,
        "F0_range": f0_range,
        "Energy_ mean": float(np.mean(rms)),
        "Energy_ std": float(np.std(rms)),
        "ZCR_mean": float(np.mean(zcr)),
        "ZCR_std": float(np.std(zcr)),
        "Spectral_centroid_mean": float(np.mean(cent)),
        "Spectral_centroid_std": float(np.std(cent)),
        "Spectral_flux_mean": _spectral_flux_mean(y, sr),
    }
    for i in range(13):
        row[f"MFCC_C{i}_mean"] = float(np.mean(mfcc[i]))
        row[f"MFCC_C{i}_std"] = float(np.std(mfcc[i]))
    for i in range(6):
        row[f"Delta_MFCC_C{i}_mean"] = float(np.mean(d06[i]))
        row[f"Delta_MFCC_C{i}_std"] = float(np.std(d06[i]))
    return row


def _load_done_set(partial_path: Path) -> set[str]:
    if not partial_path.is_file() or partial_path.stat().st_size == 0:
        return set()
    try:
        df = pd.read_csv(partial_path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return set()
    if SOURCE_COL not in df.columns:
        return set()
    return set(df[SOURCE_COL].astype(str))


def _append_checkpoint_row(partial_path: Path, row: dict[str, str | float]) -> None:
    write_header = not partial_path.is_file() or partial_path.stat().st_size == 0
    out = pd.DataFrame([row], columns=PARTIAL_COLUMNS)
    with open(partial_path, "a", newline="", encoding="utf-8") as f:
        out.to_csv(f, header=write_header, index=False)
        f.flush()


def _finalize_from_partial(
    partial_path: Path,
    out_csv: Path,
    anger_dir: Path,
    paths: list[Path],
) -> None:
    df = pd.read_csv(partial_path)
    if SOURCE_COL not in df.columns:
        raise ValueError(f"File partial thiếu cột {SOURCE_COL}: {partial_path}")
    expected = {_rel_key(p, anger_dir) for p in paths}
    df = df[df[SOURCE_COL].astype(str).isin(expected)].drop_duplicates(
        subset=[SOURCE_COL], keep="last"
    )
    if len(df) != len(paths):
        raise RuntimeError(
            f"Số dòng partial ({len(df)}) khác số file hiện tại ({len(paths)}). "
            f"Xóa {partial_path.name} nếu bạn đã đổi bộ dữ liệu."
        )
    key_order = {k: i for i, k in enumerate(sorted(expected))}
    df = df.assign(_ord=df[SOURCE_COL].map(key_order)).sort_values("_ord").drop(
        columns=["_ord"]
    )
    df[CSV_COLUMNS].to_csv(out_csv, index=False)
    partial_path.unlink(missing_ok=True)


def _iter_paths_with_progress(
    todo: list[Path],
    *,
    initial: int,
    total: int,
) -> Iterable[Path]:
    """tqdm nếu cài; không thì in [đã_xong+i/tổng] ra stderr."""
    n_todo = len(todo)
    if n_todo == 0:
        return

    try:
        from tqdm import tqdm

        yield from tqdm(
            todo,
            desc="Trích đặc trưng (anger)",
            unit="file",
            file=sys.stderr,
            mininterval=0.3,
            initial=initial,
            total=total,
        )
        return
    except ImportError:
        pass

    t0 = time.perf_counter()
    for i, p in enumerate(todo, 1):
        cur = initial + i
        elapsed = time.perf_counter() - t0
        eta_s = (elapsed / i) * (n_todo - i) if i else 0.0
        name = p.name if len(p.name) <= 44 else p.name[:41] + "..."
        sys.stderr.write(
            f"\r[{cur}/{total}] ({100 * cur / total:.1f}%) {name}  "
            f"ETA ~{eta_s / 60:.1f} phút   "
        )
        sys.stderr.flush()
        yield p
    sys.stderr.write("\n")


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    anger_dir = project_root / "Voice_Emotion_Dataset" / "anger"
    out_csv = Path(__file__).resolve().parent / "anger.csv"
    partial_path = Path(__file__).resolve().parent / PARTIAL_NAME

    if not anger_dir.is_dir():
        raise FileNotFoundError(f"Không thấy thư mục: {anger_dir}")

    patterns = ["*.wav", "*.WAV", "*.mp3", "*.flac", "*.ogg"]
    paths: list[Path] = []
    for pat in patterns:
        paths.extend(Path(p) for p in glob.glob(str(anger_dir / pat)))
    paths = sorted(set(paths))

    if not paths:
        raise FileNotFoundError(f"Không có file âm thanh trong: {anger_dir}")

    done = _load_done_set(partial_path)
    rel_for_paths = {_rel_key(p, anger_dir) for p in paths}
    done = done & rel_for_paths
    todo = [p for p in paths if _rel_key(p, anger_dir) not in done]

    n_done = len(done)
    total = len(paths)
    print(
        f"Tìm thấy {total} file; đã có trong checkpoint: {n_done}; còn xử lý: {len(todo)}",
        file=sys.stderr,
    )

    if not todo:
        if n_done == total:
            _finalize_from_partial(partial_path, out_csv, anger_dir, paths)
            print(f"Đã ghi {total} dòng -> {out_csv}")
        return

    for p in _iter_paths_with_progress(todo, initial=n_done, total=total):
        key = _rel_key(p, anger_dir)
        feats = extract_row(p)
        row: dict[str, str | float] = {SOURCE_COL: key, **feats}
        _append_checkpoint_row(partial_path, row)

    _finalize_from_partial(partial_path, out_csv, anger_dir, paths)
    print(f"Đã ghi {total} dòng -> {out_csv}")


if __name__ == "__main__":
    main()
