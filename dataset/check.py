"""
Validate a CSV against the fields listed in dataset/README.md and report missing values (NaN/empty).

Input:
  - Run with no args: file picker (Tkinter).
  - Or: python check_csv_fields.py --csv path/to/file.csv

Progress for large files: tqdm if installed; otherwise percentage on stderr.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable, TextIO

import numpy as np
import pandas as pd

_DEBUG_LOG_PATH = Path(__file__).resolve().parent.parent / "debug-7ac58c.log"
_DEBUG_SESSION = "7ac58c"


def _agent_log(
    hypothesis_id: str, location: str, message: str, data: dict
) -> None:
    # region agent log
    try:
        line = json.dumps(
            {
                "sessionId": _DEBUG_SESSION,
                "hypothesisId": hypothesis_id,
                "location": location,
                "message": message,
                "data": data,
                "timestamp": int(time.time() * 1000),
            },
            ensure_ascii=False,
        )
        with open(_DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass
    # endregion agent log


def _stream_enc(stream: TextIO) -> str | None:
    enc = getattr(stream, "encoding", None)
    return enc


def _ensure_utf8_stdio() -> None:
    """Avoid UnicodeEncodeError on Windows consoles that default to cp1252."""
    # region agent log
    _agent_log(
        "A",
        "check_csv_fields.py:_ensure_utf8_stdio",
        "stdio encoding before reconfigure",
        {
            "stdout": _stream_enc(sys.stdout),
            "stderr": _stream_enc(sys.stderr),
        },
    )
    # endregion agent log
    for stream in (sys.stdout, sys.stderr):
        try:
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError, AttributeError) as e:
            # region agent log
            _agent_log(
                "C",
                "check_csv_fields.py:_ensure_utf8_stdio",
                "reconfigure failed for stream",
                {"stream": type(stream).__name__, "error": repr(e)},
            )
            # endregion agent log
    # region agent log
    _agent_log(
        "A",
        "check_csv_fields.py:_ensure_utf8_stdio",
        "stdio encoding after reconfigure",
        {
            "stdout": _stream_enc(sys.stdout),
            "stderr": _stream_enc(sys.stderr),
        },
    )
    # endregion agent log


# Required columns — must match dataset/README.md ("CSV File Contain")
REQUIRED_COLUMNS: list[str] = [
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

REQUIRED_SET = set(REQUIRED_COLUMNS)
CHUNKSIZE = 50_000


def _pick_csv_path() -> Path | None:
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="Select CSV file to validate",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
        )
        root.destroy()
    except Exception as e:
        print(f"Could not open file picker: {e}", file=sys.stderr)
        return None
    if not path:
        return None
    return Path(path)


def _iter_chunks_with_progress(
    csv_path: Path,
    chunksize: int,
    *,
    total_rows: int | None,
) -> Iterable[pd.DataFrame]:
    reader = pd.read_csv(csv_path, chunksize=chunksize)
    try:
        from tqdm import tqdm

        if total_rows is not None and total_rows > 0:
            with tqdm(
                total=total_rows,
                desc="Reading & validating",
                unit="rows",
                file=sys.stderr,
                mininterval=0.2,
            ) as bar:
                for chunk in reader:
                    yield chunk
                    bar.update(len(chunk))
        else:
            with tqdm(
                desc="Reading & validating",
                unit="rows",
                file=sys.stderr,
                mininterval=0.2,
            ) as bar:
                for chunk in reader:
                    yield chunk
                    bar.update(len(chunk))
        return
    except ImportError:
        pass

    processed = 0
    for chunk in reader:
        yield chunk
        processed += len(chunk)
        if total_rows and total_rows > 0:
            pct = 100.0 * min(processed, total_rows) / total_rows
            sys.stderr.write(
                f"\rProcessed: {processed}/{total_rows} rows ({pct:.1f}%)   "
            )
        else:
            sys.stderr.write(f"\rProcessed: {processed} rows   ")
        sys.stderr.flush()
    sys.stderr.write("\n")


def _count_lines_fast(path: Path) -> int | None:
    """Count data rows (excluding header); None on error."""
    try:
        n = 0
        with open(path, "rb") as f:
            first = True
            for _ in f:
                if first:
                    first = False
                    continue
                n += 1
        return n
    except OSError:
        return None


def validate_csv(csv_path: Path, chunksize: int = CHUNKSIZE) -> int:
    if not csv_path.is_file():
        print(f"Error: file not found: {csv_path}")
        return 2

    header_df = pd.read_csv(csv_path, nrows=0)
    cols = list(header_df.columns)
    col_set = set(cols)

    missing = sorted(REQUIRED_SET - col_set)
    extra = sorted(col_set - REQUIRED_SET)

    print("=== Column check (per README.md) ===")
    if missing:
        print(f"Missing {len(missing)} required column(s):")
        for c in missing:
            print(f"  - {c}")
    else:
        print("All required columns are present.")

    if extra:
        print(
            f"\n{len(extra)} column(s) not listed in README (may be acceptable):"
        )
        for c in extra:
            print(f"  + {c}")

    if missing:
        print("\nSkipping NaN scan because required columns are missing.")
        return 1

    total_rows = _count_lines_fast(csv_path)
    if total_rows is not None:
        print(f"\nEstimated data rows: {total_rows}")
    else:
        print("\nCould not pre-count rows; scanning with chunked progress.")

    nan_counts: dict[str, int] = {c: 0 for c in REQUIRED_COLUMNS}
    inf_cols: set[str] = set()
    n_rows = 0

    print("\n=== Missing values (NaN / empty) and inf (single pass) ===")
    for chunk in _iter_chunks_with_progress(
        csv_path, chunksize, total_rows=total_rows
    ):
        n_rows += len(chunk)
        sub = chunk[REQUIRED_COLUMNS]
        for c in REQUIRED_COLUMNS:
            s = sub[c]
            nan_here = int(s.isna().sum())
            if pd.api.types.is_numeric_dtype(s):
                nan_counts[c] += nan_here
                v = pd.to_numeric(s, errors="coerce").to_numpy(
                    dtype=np.float64, copy=False
                )
                if np.isinf(v).any():
                    inf_cols.add(c)
            else:
                str_s = s.astype(str)
                empty = (str_s.str.strip() == "") | (str_s == "nan")
                nan_counts[c] += nan_here + int(empty.sum())

    print(f"Total rows read: {n_rows}")

    bad_cols = [(c, nan_counts[c]) for c in REQUIRED_COLUMNS if nan_counts[c] > 0]
    if not bad_cols:
        print("No missing values in required columns.")
    else:
        print("Columns with missing values:")
        for c, k in sorted(bad_cols, key=lambda x: -x[1]):
            pct = 100.0 * k / n_rows if n_rows else 0.0
            print(f"  - {c}: {k} cells ({pct:.2f}%)")

    print("\n=== Infinity check ===")
    if not inf_cols:
        print("No inf values in numeric columns.")
    else:
        for c in sorted(inf_cols):
            print(f"  - Column {c} contains inf")

    return 0 if not missing and not bad_cols and not inf_cols else 1


def main() -> int:
    _ensure_utf8_stdio()
    parser = argparse.ArgumentParser(
        description="Validate a CSV against the field list in dataset/README.md"
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Path to CSV file (omit to use the file picker dialog)",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=CHUNKSIZE,
        help=f"Rows per chunk when reading (default {CHUNKSIZE})",
    )
    args = parser.parse_args()

    path = args.csv
    if path is None:
        path = _pick_csv_path()
        if path is None:
            print("No file selected. Exiting.")
            return 2

    path = path.resolve()
    return validate_csv(path, chunksize=args.chunksize)


if __name__ == "__main__":
    sys.exit(main())