# `training` directory

## Pinned library versions

| Library | Version | pip package |
|---------|---------|-------------|
| scikit-learn (sklearn) | 1.8.0 | `scikit-learn` |
| NumPy | 2.4.4 | `numpy` |
| pandas | 3.0.2 | `pandas` |
| Matplotlib | 3.10.8 | `matplotlib` |
| Seaborn | 0.13.2 | `seaborn` |
| Optuna | 4.8.0 | `optuna` |
| SciPy | 1.17.1 | `scipy` |

`requirements.txt` also includes **IPython**, **ipykernel**, and **Jupyter** for running `.ipynb` notebooks (minimum versions, not patch-pinned).

## Virtual environment (recommended)

**Windows (PowerShell)** — from the project root `Emotion_detecting`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## Install all dependencies

From the `training` folder (or pass the full path to the file):

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Example (Windows, absolute path):

```powershell
pip install -r "d:\Code\Python\Emotion_detecting\training\requirements.txt"
```

## Verify installed versions

```bash
python -c "import sklearn, numpy, pandas, matplotlib, seaborn, optuna, scipy; print('sklearn', sklearn.__version__); print('numpy', numpy.__version__); print('pandas', pandas.__version__); print('matplotlib', matplotlib.__version__); print('seaborn', seaborn.__version__); print('optuna', optuna.__version__); print('scipy', scipy.__version__)"
```


## Launch Jupyter

```bash
jupyter notebook
```

You can also use the Jupyter / notebook integration in VS Code or Cursor and select the Python interpreter (or kernel) where you installed these packages.

## Troubleshooting

- **`sklearn` not found on pip:** install **`scikit-learn`** (already listed in `requirements.txt`).
- **Version conflicts:** create a fresh venv, then only `pip install -r requirements.txt` inside it.
- **Python too old:** NumPy 2.x / pandas 3.x typically need a recent Python (e.g. 3.12+ or whatever each package documents at install time).

## Notes

- Example data for the notebooks is usually at `../dataset/all_emotions.csv` (see `RF.md` and the notebooks).
