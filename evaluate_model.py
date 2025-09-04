# ===================================
# EVALUASI MODEL BPNN
# ===================================

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -----------------------------------
# 1. Load dataset hasil ekstraksi fitur
# -----------------------------------
df = pd.read_csv('dataset/keriput_dataset.csv')

# Pastikan semua fitur yang dibutuhkan tersedia
required_features = ['dahi', 'mata', 'pipi', 'mulut', 'jumlah_kontur', 'panjang_total_kontur']
for col in required_features:
    if col not in df.columns:
        raise ValueError(f"Kolom '{col}' tidak ditemukan di dataset. Pastikan ekstraksi fitur sudah benar.")

# Pisahkan fitur (X) dan label (y)
X = df[required_features].values
y_true = df['label'].values

# -----------------------------------
# 2. Load scaler (normalisasi) jika tersedia
# -----------------------------------
try:
    scaler = joblib.load('model/scaler.pkl')
    X_scaled = scaler.transform(X)
except:
    scaler = None
    X_scaled = X

# -----------------------------------
# 3. Load model BPNN hasil training
# -----------------------------------
model = joblib.load('model/bpnn_model.pkl')

# -----------------------------------
# 4. Load label encoder jika digunakan saat training
# -----------------------------------
try:
    label_encoder = joblib.load('model/label_encoder.pkl')
    y_true_enc = label_encoder.transform(y_true)
except:
    label_encoder = None
    y_true_enc = y_true

# -----------------------------------
# 5. Prediksi dengan model
# -----------------------------------
y_pred = model.predict(X_scaled)

# Jika label encoder ada, kembalikan hasil prediksi ke bentuk label asli
if label_encoder is not None:
    y_pred_label = label_encoder.inverse_transform(y_pred)
    y_true_label = y_true
else:
    y_pred_label = y_pred
    y_true_label = y_true

# -----------------------------------
# 6. Evaluasi model: Akurasi, Confusion Matrix, dan Classification Report
# -----------------------------------
print('Akurasi:', accuracy_score(y_true_label, y_pred_label))
print('\nConfusion Matrix:')
print(confusion_matrix(y_true_label, y_pred_label))
print('\nClassification Report (precision, recall, f1-score):')
print(classification_report(y_true_label, y_pred_label, digits=4))
