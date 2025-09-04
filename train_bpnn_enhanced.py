# ================================
# TRAINING MODEL BPNN
# ================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# --------------------------------
# Fungsi: Load dataset train & test, scaling, dan encoding label
# --------------------------------
def load_and_prepare_data():
    """Load and prepare data for training."""
    
    # Membaca dataset train dan test
    train_data = pd.read_csv("dataset/keriput_train.csv")
    test_data = pd.read_csv("dataset/keriput_test.csv")
    
    # Tampilkan distribusi label
    print('Distribusi label di dataset training:')
    print(train_data['label'].value_counts())
    print('\nDistribusi label di dataset testing:')
    print(test_data['label'].value_counts())
    
    # Tentukan fitur yang digunakan
    fitur_names = ['dahi', 'mata', 'pipi', 'mulut']
    if 'jumlah_kontur' in train_data.columns and 'panjang_total_kontur' in train_data.columns:
        fitur_names += ['jumlah_kontur', 'panjang_total_kontur']
    
    print(f"\nFitur yang digunakan: {fitur_names}")
    
    # Data training + augmentasi dengan menambahkan noise kecil
    X_train = []
    y_train = []
    for idx, row in train_data.iterrows():
        fitur = [row[nama] for nama in fitur_names]
        label = row['label']

        X_train.append(fitur)
        y_train.append(label)

        for _ in range(3):  
            augmented = [f + np.random.normal(0, 0.01) for f in fitur]
            X_train.append(augmented)
            y_train.append(label)

    # Data testing
    X_test = []
    y_test = []
    for idx, row in test_data.iterrows():
        fitur = [row[nama] for nama in fitur_names]
        label = row['label']
        X_test.append(fitur)
        y_test.append(label)

    # Encode label ke bentuk numerik
    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_test_enc = label_encoder.transform(y_test)

    # Konversi ke numpy array
    X_train = np.array(X_train)
    X_test = np.array(X_test)

    # Normalisasi fitur dengan StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train_enc, y_test_enc, label_encoder, scaler, fitur_names

# --------------------------------
# Fungsi: Uji berbagai konfigurasi hidden layer MLPClassifier
# --------------------------------
def test_hidden_layer_configurations():
    """Test various hidden layer configurations."""
    
    # Daftar konfigurasi hidden layer yang diuji
    configurations = [
        (8, 8), (16, 16), (32, 32), (64, 64), (128, 128),
        (16, 8), (32, 8), (64, 8), (128, 8),
        (8, 16), (16, 32), (32, 64), (64, 128),
        (32, 16), (64, 32), (128, 64),
        (8, 32), (16, 64), (32, 128),
        (8, 4), (16, 4), (32, 4), (64, 4),
        (4, 8), (4, 16), (4, 32), (4, 64)
    ]
    
    print(f"Total konfigurasi yang akan diuji: {len(configurations)}")
    
    # Load data
    X_train, X_test, y_train, y_test, label_encoder, scaler, feature_names = load_and_prepare_data()
    
    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Buat folder output untuk hasil eksperimen
    os.makedirs("output", exist_ok=True)
    os.makedirs(f"output/experiments_{timestamp}", exist_ok=True)
    
    # Loop semua konfigurasi hidden layer
    for idx, hidden_layers in enumerate(configurations):
        print(f"\n{'='*60}")
        print(f"Menguji konfigurasi {idx+1}/{len(configurations)}: {hidden_layers}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            # Inisialisasi model MLP
            mlp = MLPClassifier(
                hidden_layer_sizes=hidden_layers,
                max_iter=1000,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.2,
                n_iter_no_change=50,
                verbose=False 
            )
            
            # Training model
            mlp.fit(X_train, y_train)
            
            # Prediksi pada data test
            y_pred = mlp.predict(X_test)
            
            # Hitung akurasi
            accuracy = accuracy_score(y_test, y_pred)
            training_time = time.time() - start_time
            
            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            
            # Simpan plot confusion matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=label_encoder.classes_,
                        yticklabels=label_encoder.classes_)
            plt.title(f"Confusion Matrix - Hidden Layers: {hidden_layers}\nAccuracy: {accuracy:.4f}")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.tight_layout()
            plt.savefig(f"output/experiments_{timestamp}/confusion_matrix_{hidden_layers}.png")
            plt.close()
            
            # Simpan kurva loss training
            plt.figure(figsize=(10, 5))
            plt.plot(mlp.loss_curve_)
            plt.title(f"Training Loss - Hidden Layers: {hidden_layers}")
            plt.xlabel("Iterations")
            plt.ylabel("Loss")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"output/experiments_{timestamp}/training_loss_{hidden_layers}.png")
            plt.close()
            
            # Simpan hasil
            result = {
                'configuration': str(hidden_layers),
                'hidden_layers': hidden_layers,
                'accuracy': accuracy,
                'training_time': training_time,
                'iterations': len(mlp.loss_curve_),
                'final_loss': mlp.loss_curve_[-1] if mlp.loss_curve_ else None
            }
            
            results.append(result)
            
            print(f"Akurasi: {accuracy:.4f}")
            print(f"Waktu training: {training_time:.2f} detik")
            print(f"Iterasi: {result['iterations']}")
            print(f"Loss akhir: {result['final_loss']:.4f}" if result['final_loss'] else "Loss akhir: N/A")
            
            # Laporan klasifikasi
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
            
        except Exception as e:
            print(f"Error pada konfigurasi {hidden_layers}: {str(e)}")
            result = {
                'configuration': str(hidden_layers),
                'hidden_layers': hidden_layers,
                'accuracy': 0.0,
                'training_time': 0.0,
                'iterations': 0,
                'final_loss': None,
                'error': str(e)
            }
            results.append(result)
    
    # Simpan hasil eksperimen ke CSV
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('accuracy', ascending=False)
    results_df.to_csv(f"output/experiments_{timestamp}/hidden_layer_results.csv", index=False)
    
    # Buat visualisasi ringkasan
    create_summary_visualization(results_df, timestamp)
    
    # Ambil model terbaik
    best_result = results_df.iloc[0]
    best_config = eval(best_result['configuration'])
    
    print(f"\n{'='*60}")
    print("KONFIGURASI TERBAIK DITEMUKAN!")
    print(f"{'='*60}")
    print(f"Konfigurasi terbaik: {best_config}")
    print(f"Akurasi tertinggi: {best_result['accuracy']:.4f}")
    print(f"Waktu training: {best_result['training_time']:.2f} detik")
    
    # Retrain model terbaik
    print("\nMenyimpan model terbaik...")
    best_mlp = MLPClassifier(
        hidden_layer_sizes=best_config,
        max_iter=1000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=50
    )
    
    best_mlp.fit(X_train, y_train)
    
    # Simpan model, scaler, dan label encoder
    os.makedirs("model", exist_ok=True)
    joblib.dump(best_mlp, f'model/bpnn_model_best_{timestamp}.pkl')
    joblib.dump(scaler, f'model/scaler_best_{timestamp}.pkl')
    joblib.dump(label_encoder, f'model/label_encoder_best_{timestamp}.pkl')
    
    # Simpan versi generik (untuk prediksi langsung)
    joblib.dump(best_mlp, 'model/bpnn_model.pkl')
    joblib.dump(scaler, 'model/scaler.pkl')
    joblib.dump(label_encoder, 'model/label_encoder.pkl')
    
    print(f"Model terbaik disimpan dengan timestamp: {timestamp}")
    print("Model juga disimpan sebagai bpnn_model.pkl untuk prediksi")
    
    return results_df, timestamp

# --------------------------------
# Fungsi: Buat ringkasan visualisasi hasil eksperimen
# --------------------------------
def create_summary_visualization(results_df, timestamp):
    """Create summary visualizations of all experiments."""
    
    # Plot perbandingan akurasi, waktu training, dan top 10 konfigurasi
    plt.figure(figsize=(15, 8))
    
    # Plot accuracy comparison
    plt.subplot(2, 2, 1)
    configurations = results_df['configuration'].astype(str)
    accuracies = results_df['accuracy']
    plt.bar(range(len(configurations)), accuracies)
    plt.xlabel('Configuration Index')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Comparison Across Configurations')
    plt.xticks(range(len(configurations)), configurations, rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Plot top 10 best configurations
    top_10 = results_df.head(10)
    plt.subplot(2, 2, 2)
    plt.barh(range(len(top_10)), top_10['accuracy'])
    plt.yticks(range(len(top_10)), top_10['configuration'])
    plt.xlabel('Accuracy')
    plt.title('Top 10 Best Configurations')
    plt.grid(True, alpha=0.3)
    
    # Plot training time comparison
    plt.subplot(2, 2, 3)
    plt.bar(range(len(configurations)), results_df['training_time'])
    plt.xlabel('Configuration Index')
    plt.ylabel('Training Time (seconds)')
    plt.title('Training Time Comparison')
    plt.xticks(range(len(configurations)), configurations, rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # Scatter plot Accuracy vs Training Time
    plt.subplot(2, 2, 4)
    plt.scatter(results_df['training_time'], results_df['accuracy'], 
                c=range(len(results_df)), cmap='viridis', alpha=0.7)
    plt.xlabel('Training Time (seconds)')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Training Time')
    plt.colorbar(label='Configuration Index')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"output/experiments_{timestamp}/hidden_layer_comparison_{timestamp}.png", dpi=300)
    plt.close()
    
    # Simpan laporan ringkasan eksperimen
    with open(f"output/experiments_{timestamp}/summary_report_{timestamp}.txt", 'w') as f:
        f.write("LAPORAN HASIL EKSPERIMEN BPNN\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total konfigurasi yang diuji: {len(results_df)}\n")
        f.write(f"Waktu eksperimen: {timestamp}\n\n")
        
        f.write("KONFIGURASI TERBAIK:\n")
        f.write("-" * 30 + "\n")
        best = results_df.iloc[0]
        f.write(f"Konfigurasi: {best['configuration']}\n")
        f.write(f"Akurasi: {best['accuracy']:.4f}\n")
        f.write(f"Waktu training: {best['training_time']:.2f} detik\n")
        f.write(f"Iterasi: {best['iterations']}\n\n")
        
        f.write("10 KONFIGURASI TERBAIK:\n")
        f.write("-" * 30 + "\n")
        for idx, row in results_df.head(10).iterrows():
            f.write(f"{idx+1}. {row['configuration']} - Akurasi: {row['accuracy']:.4f} - Waktu: {row['training_time']:.2f}s\n")
        
        f.write("\nSEMUA HASIL:\n")
        f.write("-" * 30 + "\n")
        for idx, row in results_df.iterrows():
            f.write(f"{row['configuration']} - Akurasi: {row['accuracy']:.4f} - Waktu: {row['training_time']:.2f}s\n")

# --------------------------------
# Main eksekusi eksperimen training BPNN
# --------------------------------
if __name__ == "__main__":
    print("Memulai eksperimen BPNN dengan berbagai konfigurasi hidden layer...")
    results_df, timestamp = test_hidden_layer_configurations()
    
    print("\n" + "="*60)
    print("EKSPERIMEN SELESAI!")
    print("="*60)
    print(f"Hasil lengkap tersedia di: output/experiments_{timestamp}/")
    print(f"Model terbaik tersimpan di: model/bpnn_model_best_{timestamp}.pkl")
    print(f"Skala terbaik tersimpan di: model/scaler_best_{timestamp}.pkl")
    print(f"Label encoder tersimpan di: model/label_encoder_best_{timestamp}.pkl")
