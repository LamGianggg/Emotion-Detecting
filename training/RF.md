# Dataset 
- Emotion_detecting\dataset\all_emotions.csv

# Library
- sklearn
- numpy
- pandas
- matplotlib
- seaborn 
- os
- sys
- optuna
- warnings
- IPython
- scipy

# Feature Col
- F0_mean
- F0_std
- F0_range
- Energy_ mean
- Energy_ std
- ZCR_mean
- ZCR_std
- Spectral_centroid_mean
- Spectral_centroid_std
- Spectral_flux_mean
- MFCC_C0_mean
- MFCC_C1_mean
- MFCC_C2_mean
- MFCC_C3_mean
- MFCC_C4_mean
- MFCC_C5_mean
- MFCC_C6_mean
- MFCC_C7_mean
- MFCC_C8_mean
- MFCC_C9_mean
- MFCC_C10_mean
- MFCC_C11_mean
- MFCC_C12_mean
- MFCC_C0_std
- MFCC_C1_std
- MFCC_C2_std
- MFCC_C3_std
- MFCC_C4_std
- MFCC_C5_std
- MFCC_C6_std
- MFCC_C7_std
- MFCC_C8_std
- MFCC_C9_std
- MFCC_C10_std
- MFCC_C11_std
- MFCC_C12_std
- Delta_MFCC_C0_mean
- Delta_MFCC_C1_mean
- Delta_MFCC_C2_mean
- Delta_MFCC_C3_mean
- Delta_MFCC_C4_mean
- Delta_MFCC_C5_mean
- Delta_MFCC_C0_std
- Delta_MFCC_C1_std
- Delta_MFCC_C2_std
- Delta_MFCC_C3_std
- Delta_MFCC_C4_std
- Delta_MFCC_C5_std

# Target Col
- Label

# Data Preprocessing
- Kiểm tra và xử lý NaN / Inf trước khi train (thay bằng median)
- Không dùng StandardScaler với Feature Col
- Train/Test split theo tỉ lệ 80/20, dùng stratify=Label (bắt buộc với multi-class)

# Random Forest Model
- Không dùng StandardScaler với Feature Col
- Dùng Optuna để tự động tìm thông số tối ưu (TPE Sampler, 60 trials, CV 5-fold stratified)
- Các thông số Optuna tune:
  - N_ESTIMATORS : 150 -> 250
  - max_depth    : 15 -> 35 + None (không giới hạn)
  - max_features : sqrt | log2 | None
  - min_samples_split : 2 -> 10
  - min_samples_leaf  : 1 -> 5
  - bootstrap    : True | False
  - class_weight : balanced | None

# Output
- Correlation heatmap (MFCC features)
- Class distribution plot
- Optuna optimization history plot (F1 mỗi trial + convergence curve)
- OOB Error Curve theo n_estimators (thay thế Loss plot — RF không có loss)
- Learning curve (train vs validation F1)
- Confusion matrix (số lượng + tỉ lệ)
- F1 Score (weighted)
- Cohen Kappa Score
- ROC curve — One-vs-Rest (OvR) cho multi-class + AUC từng lớp
- Feature importance plot (Top 25)
- Classification Report đầy đủ

# Lưu ý
- MSE / RMSE không dùng (metric của Regression, không phù hợp Classification)
- ROC curve phải dùng One-vs-Rest (OvR), không dùng binary ROC
- OOB Score chỉ tính được khi bootstrap = True

# Các cell trong ipynb
- Cell 1  : Import các thư viện
- Cell 2  : Build đường dẫn và check lại đường dẫn tới file csv
- Cell 3  : Kiểm tra null, duplicated và check tổng label có trong file csv
- Cell 4  : Chuẩn bị Feature Col và Target Col
- Cell 5  : Kiểm tra và xử lý NaN / Inf trong Feature Col (thay bằng median)
- Cell 6  : Vẽ class distribution plot — kiểm tra mất cân bằng nhãn
- Cell 7  : Vẽ correlation heatmap MFCC features
- Cell 8  : Train/Test split (80/20, stratify=Label)
- Cell 9  : Định nghĩa Optuna objective function
- Cell 10 : Chạy Optuna study — tìm thông số tối ưu, in best_params
- Cell 11 : Vẽ Optuna optimization history (F1 mỗi trial + convergence curve)
- Cell 12 : Train mô hình Random Forest với best_params
- Cell 13 : Vẽ OOB Error Curve theo n_estimators (chỉ khi bootstrap=True)
- Cell 14 : Vẽ Learning Curve (train vs validation F1)
- Cell 15 : Dự đoán trên tập test — lấy y_pred và y_pred_proba
- Cell 16 : In Classification Report + F1 Weighted + Cohen Kappa
- Cell 17 : Vẽ Confusion Matrix (số lượng + tỉ lệ)
- Cell 18 : Vẽ ROC Curve One-vs-Rest + AUC từng lớp
- Cell 19 : Vẽ Feature Importance (Top 25)
- Cell 20 : Tổng kết kết quả — in bảng tóm tắt toàn bộ metrics