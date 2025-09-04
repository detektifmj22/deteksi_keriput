#!/usr/bin/env python3
"""
Script untuk menjalankan training BPNN dengan berbagai konfigurasi hidden layer
"""

import os
import sys
from train_bpnn_enhanced import test_hidden_layer_configurations

def main():
    print("=" * 70)
    print("PROGRAM TRAINING BPNN ENHANCED")
    print("Menguji Semua Konfigurasi Hidden Layer")
    print("=" * 70)
    
    required_files = [
        "dataset/keriput_train.csv",
        "dataset/keriput_test.csv"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("ERROR: File berikut tidak ditemukan:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nPastikan dataset telah diproses terlebih dahulu.")
        return
    
    try:
        results_df, timestamp = test_hidden_layer_configurations()
        
        print("\n" + "=" * 70)
        print("RINGKASAN AKHIR")
        print("=" * 70)
        
        best_config = results_df.iloc[0]
        print(f"Konfigurasi Terbaik: {best_config['configuration']}")
        print(f"Akurasi Tertinggi: {best_config['accuracy']:.4f}")
        print(f"Waktu Training: {best_config['training_time']:.2f} detik")
        
        print(f"\nSemua hasil tersimpan di: output/experiments_{timestamp}/")
        print(f"Model terbaik: model/bpnn_model_best_{timestamp}.pkl")
        
        print("\n5 Konfigurasi Terbaik:")
        print("-" * 40)
        for i, (_, row) in enumerate(results_df.head(5).iterrows(), 1):
            print(f"{i}. {row['configuration']} - Akurasi: {row['accuracy']:.4f}")
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        print("Pastikan semua dependensi terinstal dengan benar.")

if __name__ == "__main__":
    main()
