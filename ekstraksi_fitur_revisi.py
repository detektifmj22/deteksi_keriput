# ================================
# EKSTRAKSI FITUR KERIPUT WAJAH
# ================================

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

# Fungsi untuk ekstraksi fitur GLCM pada area tertentu
def ekstrak_glcm_fitur(area):
    """Ekstraksi fitur GLCM untuk area tertentu"""
    if area.size == 0:  # Jika area kosong
        return 0.0
    
    # Konversi ke grayscale jika masih berwarna
    if len(area.shape) == 3:
        area = cv2.cvtColor(area, cv2.COLOR_BGR2GRAY)
    
    # Hitung matriks GLCM
    glcm = graycomatrix(area, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0,0]        # Kontras
    homogeneity = graycoprops(glcm, 'homogeneity')[0,0]  # Homogenitas
    energy = graycoprops(glcm, 'energy')[0,0]            # Energi
    return (contrast + homogeneity + energy) / 3  # Rata-rata fitur GLCM

# Fungsi untuk ekstraksi fitur dari seluruh area wajah
def ekstrak_fitur_keriput(img):
    """Ekstraksi fitur untuk semua area wajah yang dibutuhkan"""
    h, w = img.shape[:2]
    
    # Potong area wajah: dahi, mata, pipi, mulut
    dahi = img[30:80, 50:150]
    mata = img[70:110, 60:140]
    
    pipi_kiri = img[90:130, 30:70]
    pipi_kanan = img[90:130, 130:170]
    pipi = np.concatenate([pipi_kiri, pipi_kanan], axis=1)
    
    mulut = img[140:180, 70:130]
    
    # Hitung fitur GLCM dari setiap area
    fitur_dahi = ekstrak_glcm_fitur(dahi)
    fitur_mata = ekstrak_glcm_fitur(mata)
    fitur_pipi = ekstrak_glcm_fitur(pipi)
    fitur_mulut = ekstrak_glcm_fitur(mulut)
    
    # Deteksi tepi untuk menghitung jumlah dan panjang kontur
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)
    edges = cv2.Canny(img_blur, 30, 80)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    jumlah_kontur = len(contours)  # Jumlah kontur
    panjang_total_kontur = sum([cv2.arcLength(cnt, False) for cnt in contours])  # Panjang total kontur

    # Return fitur dalam bentuk list
    return [fitur_dahi, fitur_mata, fitur_pipi, fitur_mulut, jumlah_kontur, panjang_total_kontur]

# Import fungsi preprocessing untuk memuat dan memproses citra
from preprocessing import load_and_preprocess_images

# Eksekusi utama jika file dijalankan langsung
if __name__ == "__main__":
    import os
    import pandas as pd

    data = []  # List untuk menyimpan fitur
    base_dir = "dataset"
    kelas_list = ["high", "medium"]  # Dua kelas keriput
    
    # Looping tiap kelas
    for kelas in kelas_list:
        folder = os.path.join(base_dir, kelas)
        label = kelas
        images, filenames = load_and_preprocess_images(folder)  # Load dataset
        for img, filename in zip(images, filenames):
            fitur = ekstrak_fitur_keriput(img)  # Ekstrak fitur
            data.append({
                'dahi': fitur[0],
                'mata': fitur[1],
                'pipi': fitur[2],
                'mulut': fitur[3],
                'jumlah_kontur': fitur[4],
                'panjang_total_kontur': fitur[5],
                'label': label
            })

    # Simpan hasil ekstraksi ke file CSV
    os.makedirs(base_dir, exist_ok=True)
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(base_dir, "keriput_dataset.csv"), index=False)
    print(f"Ekstraksi selesai. Data disimpan di {os.path.join(base_dir, 'keriput_dataset.csv')}")
