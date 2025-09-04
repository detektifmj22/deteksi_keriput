# ==========================================
# UTILITAS UNTUK LOAD & PREDIKSI DENGAN MODEL BPNN
# ==========================================

import joblib
import os
import numpy as np

# ------------------------------------------
# Fungsi untuk load model, scaler, dan label encoder
# ------------------------------------------
def load_model_components():
    """
    Load model, scaler, dan label encoder dengan fallback mechanism.
    Jika bpnn_model.pkl tidak ada, otomatis mencari model terbaik dengan timestamp.
    
    Returns:
        tuple: (model, scaler, label_encoder)
    """
    try:
        # Coba load model default
        model = joblib.load('model/bpnn_model.pkl')
        scaler = joblib.load('model/scaler.pkl')
        label_encoder = joblib.load('model/label_encoder.pkl')
        print("Model berhasil dimuat dari model/bpnn_model.pkl")
        return model, scaler, label_encoder
    except FileNotFoundError:
        # Jika tidak ada, cari model terbaik berdasarkan timestamp
        model_files = [f for f in os.listdir('model') if f.startswith('bpnn_model_best_') and f.endswith('.pkl')]
        if model_files:
            latest_model = sorted(model_files)[-1]  # Ambil model terbaru
            latest_timestamp = latest_model.replace('bpnn_model_best_', '').replace('.pkl', '')
            
            model = joblib.load(f'model/{latest_model}')
            scaler = joblib.load(f'model/scaler_best_{latest_timestamp}.pkl')
            label_encoder = joblib.load(f'model/label_encoder_best_{latest_timestamp}.pkl')
            print(f"Model terbaik berhasil dimuat: {latest_model}")
            return model, scaler, label_encoder
        else:
            raise FileNotFoundError("Tidak ada model yang ditemukan di folder model/")

# ------------------------------------------
# Fungsi untuk prediksi dengan model
# ------------------------------------------
def predict_with_model(features):
    """
    Melakukan prediksi menggunakan model BPNN yang sudah diload.
    
    Args:
        features: array fitur (bisa 1 sampel atau banyak sampel)
        
    Returns:
        predictions: hasil prediksi dalam bentuk label asli
    """
    model, scaler, label_encoder = load_model_components()
    
    # Jika input hanya 1 sampel (1D array), reshape ke 2D
    features_scaled = scaler.transform(
        np.array(features).reshape(1, -1) if len(np.array(features).shape) == 1 else features
    )
    
    # Prediksi label encoded
    predictions_encoded = model.predict(features_scaled)
    # Decode ke label asli (misalnya "high", "medium")
    predictions = label_encoder.inverse_transform(predictions_encoded)
    
    return predictions

# ------------------------------------------
# Fungsi untuk melihat model apa saja yang tersedia
# ------------------------------------------
def get_model_info():
    """
    Menampilkan daftar file model (.pkl) yang ada di folder 'model'.
    
    Returns:
        list: daftar nama file model
    """
    model_files = [f for f in os.listdir('model') if f.endswith('.pkl')]
    print("Model yang tersedia:")
    for f in sorted(model_files):
        print(f"  - {f}")
    
    return model_files
