# Thư mục `training` — cài đặt môi trường

File `requirements.txt` trong thư mục này ghim các phiên bản thư viện dùng cho notebook (ví dụ `random_forest.ipynb`) và các script liên quan.

## Phiên bản đã ghim

| Thư viện   | Phiên bản | Gói pip        |
|-----------|-----------|----------------|
| scikit-learn (sklearn) | 1.8.0 | `scikit-learn` |
| NumPy     | 2.4.4     | `numpy`        |
| pandas    | 3.0.2     | `pandas`       |
| Matplotlib| 3.10.8    | `matplotlib`   |
| Seaborn   | 0.13.2    | `seaborn`      |
| Optuna    | 4.8.0     | `optuna`       |
| SciPy     | 1.17.1    | `scipy`        |

Ngoài ra có thêm **IPython**, **ipykernel**, **jupyter** để mở và chạy file `.ipynb` (không ghim patch version để linh hoạt với môi trường của bạn).

## Chuẩn bị (khuyến nghị: virtual environment)

**Windows (PowerShell)** — từ thư mục gốc dự án `Emotion_detecting`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## Cài đặt tất cả thư viện

Từ thư mục `training` (hoặc chỉ định đường dẫn đầy đủ tới file):

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Hoặc một dòng với đường dẫn tuyệt đối (Windows):

```powershell
pip install -r "d:\Code\Python\Emotion_detecting\training\requirements.txt"
```

## Kiểm tra phiên bản đã cài

```bash
python -c "import sklearn, numpy, pandas, matplotlib, seaborn, optuna, scipy; print('sklearn', sklearn.__version__); print('numpy', numpy.__version__); print('pandas', pandas.__version__); print('matplotlib', matplotlib.__version__); print('seaborn', seaborn.__version__); print('optuna', optuna.__version__); print('scipy', scipy.__version__)"
```

Hoặc:

```bash
pip show scikit-learn numpy pandas matplotlib seaborn optuna scipy
```

## Mở Jupyter

```bash
jupyter notebook
```

hoặc dùng Jupyter trong VS Code / Cursor: chọn kernel Python trỏ tới `.venv` (hoặc interpreter đã cài `ipykernel`).

## Xử lý lỗi thường gặp

- **`sklearn` không tìm thấy trên pip:** cài bằng tên **`scikit-learn`** (đã có trong `requirements.txt`).
- **Xung đột phiên bản:** tạo venv mới, chỉ cài `requirements.txt` trong venv đó rồi thử lại.
- **Python quá cũ:** các bản NumPy 2.x / pandas 3.x thường cần Python mới (khuyến nghị Python 3.11+ hoặc theo tài liệu từng gói tại thời điểm bạn cài).

## Ghi chú

- Dữ liệu mẫu cho notebook thường nằm ở `../dataset/all_emotions.csv` (xem `RF.md` / notebook).
- Nếu bạn thêm notebook hoặc script cần thư viện khác (ví dụ `torch`), hãy bổ sung vào `requirements.txt` và cập nhật bảng trong README này.
