# ===================================
# PREDIKSI KERIPUT DARI CITRA
# ===================================

from preprocessing import load_and_preprocess_images_from_csv
from ekstraksi_fitur_revisi import ekstrak_fitur_keriput
import joblib
import pandas as pd
import numpy as np
import os

# -----------------------------------
# 1. Load gambar & label dari CSV
# -----------------------------------
images, filenames, labels = load_and_preprocess_images_from_csv('dataset/keriput_dataset/')

# Ekstraksi fitur untuk setiap gambar
fitur_data = [ekstrak_fitur_keriput(img) for img in images]

# -----------------------------------
# 2. Load model, scaler, dan label encoder
# -----------------------------------
try:
    # Coba load model default
    model = joblib.load('model/bpnn_model.pkl')
    scaler = joblib.load('model/scaler.pkl')
    label_encoder = joblib.load('model/label_encoder.pkl')
    print("Model berhasil dimuat dari model/bpnn_model.pkl")
except FileNotFoundError:
    # Jika model default tidak ada, cari model terbaik (best) berdasarkan timestamp
    model_files = [f for f in os.listdir('model') if f.startswith('bpnn_model_best_') and f.endswith('.pkl')]
    if model_files:
        latest_model = sorted(model_files)[-1]  # Ambil model terbaru
        latest_timestamp = latest_model.replace('bpnn_model_best_', '').replace('.pkl', '')
        
        model = joblib.load(f'model/{latest_model}')
        scaler = joblib.load(f'model/scaler_best_{latest_timestamp}.pkl')
        label_encoder = joblib.load(f'model/label_encoder_best_{latest_timestamp}.pkl')
        print(f"Model terbaik berhasil dimuat: {latest_model}")
    else:
        raise FileNotFoundError("Tidak ada model yang ditemukan di folder model/")

# -----------------------------------
# 3. Preprocessing fitur
# -----------------------------------
fitur_array = np.array(fitur_data)
fitur_scaled = scaler.transform(fitur_array)

# -----------------------------------
# 4. Prediksi keriput dengan model
# -----------------------------------
prediksi_encoded = model.predict(fitur_scaled)
prediksi = label_encoder.inverse_transform(prediksi_encoded)

# -----------------------------------
# 5. Simpan hasil prediksi ke CSV
# -----------------------------------
df_hasil = pd.DataFrame(fitur_data, columns=["dahi", "mata", "pipi", "mulut", "jumlah_kontur", "panjang_total_kontur"])
df_hasil['prediksi_keriput'] = prediksi
df_hasil['nama_file'] = filenames

os.makedirs('output', exist_ok=True)
df_hasil.to_csv('output/hasil_prediksi.csv', index=False)

# -----------------------------------
# 6. Informasi hasil prediksi
# -----------------------------------
print("Hasil prediksi disimpan di output/hasil_prediksi.csv")
print(f"Total gambar diproses: {len(images)}")
print(f"Distribusi prediksi: {df_hasil['prediksi_keriput'].value_counts()}")
