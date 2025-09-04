# ================================
# PREPROCESSING DATASET KERIPUT
# ================================

import pandas as pd

# Path file CSV hasil ekstraksi fitur
csv_path = "d:\\deteksi_keriput\\dataset\\keriput_dataset.csv"

# Fungsi untuk memuat dataset dari CSV
def load_dataset_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    
    # Ambil kolom fitur
    X = df[["dahi", "mata", "pipi", "mulut", "jumlah_kontur", "panjang_total_kontur"]].values
    # Ambil kolom label
    y = df["label"].values
    
    return X, y

# Load dataset dan tampilkan informasi awal
X, y = load_dataset_from_csv(csv_path)
print(f"Jumlah data: {len(X)}")
print(f"Label unik: {set(y)}")
print("5 data pertama:")
print(X[:5], y[:5])

# Fungsi untuk load dataset sekaligus melakukan preprocessing numerik
def load_and_preprocess_images(folder):
    """
    Load dataset from CSV dan lakukan preprocessing sederhana pada kolom numerik.
    Return X (fitur) dan y (label).
    """
    # Saat ini parameter folder tidak digunakan, path CSV sudah fixed
    csv_path = "d:\\deteksi_keriput\\dataset\\keriput_dataset.csv"
    df = load_dataset_from_csv(csv_path)  # ⚠️ df di sini sebenarnya berupa tuple (X, y), bukan DataFrame
    if df is None or df.empty:
        print("Dataset kosong atau tidak ditemukan.")
        return None, None

    # Import scaler untuk normalisasi
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()

    # Pilih kolom numerik
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if not numeric_cols.empty:
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    # Pisahkan fitur dan label jika ada kolom label
    if 'label' in df.columns:
        X = df.drop(columns=['label'], errors='ignore')
        y = df['label']
    else:
        X = df
        y = None

    return X, y
