# Thư mục `training` 

## Phiên bản thư viện 

| Thư viện   | Phiên bản | Gói pip        |
|-----------|-----------|----------------|
| scikit-learn (sklearn) | 1.8.0 | `scikit-learn` |
| NumPy     | 2.4.4     | `numpy`        |
| pandas    | 3.0.2     | `pandas`       |
| Matplotlib| 3.10.8    | `matplotlib`   |
| Seaborn   | 0.13.2    | `seaborn`      |
| Optuna    | 4.8.0     | `optuna`       |
| SciPy     | 1.17.1    | `scipy`        |


## Cài đặt tất cả thư viện

Từ thư mục `training` (hoặc chỉ định đường dẫn đầy đủ tới file):

```bash
pip install --upgrade pip
pip install -r requirements.txt
```


## Kiểm tra phiên bản đã cài

```bash
python -c "import sklearn, numpy, pandas, matplotlib, seaborn, optuna, scipy; print('sklearn', sklearn.__version__); print('numpy', numpy.__version__); print('pandas', pandas.__version__); print('matplotlib', matplotlib.__version__); print('seaborn', seaborn.__version__); print('optuna', optuna.__version__); print('scipy', scipy.__version__)"
```


## Mở Jupyter

```bash
jupyter notebook
```

hoặc dùng Jupyter trong VS Code / Cursor

## Xử lý lỗi thường gặp

- **`sklearn` không tìm thấy trên pip:** cài bằng tên **`scikit-learn`** (đã có trong `requirements.txt`).
- **Xung đột phiên bản:** tạo venv mới, chỉ cài `requirements.txt` trong venv đó rồi thử lại.
- **Python quá cũ:** các bản NumPy 2.x / pandas 3.x thường cần Python mới (khuyến nghị Python 3.12 hoặc theo tài liệu từng gói tại thời điểm bạn cài).

## Ghi chú

- Dữ liệu mẫu cho notebook thường nằm ở `../dataset/all_emotions.csv` (xem `RF.md` / notebook).

