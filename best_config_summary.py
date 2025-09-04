import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Read the combined results
df = pd.read_csv('output/all_configuration_results.csv')

# Get best configuration
best_config = df.iloc[0]

# Create a focused summary visualization
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Ringkasan Konfigurasi Terbaik Model Deteksi Keriput', fontsize=16, fontweight='bold')

# 1. Best Configuration Highlight
config_str = str(best_config['configuration']).replace(',', ', ')
ax1.text(0.5, 0.7, f'Konfigurasi Terbaik:\n{config_str}', 
         ha='center', va='center', fontsize=14, fontweight='bold',
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", edgecolor="navy"))

ax1.text(0.5, 0.4, f'Akurasi: {best_config["accuracy"]:.4f}', 
         ha='center', va='center', fontsize=12, color='green')

ax1.text(0.5, 0.3, f'Waktu Training: {best_config["training_time"]:.2f} detik', 
         ha='center', va='center', fontsize=12, color='orange')

ax1.text(0.5, 0.2, f'Iterasi: {best_config["iterations"]}', 
         ha='center', va='center', fontsize=12, color='purple')

ax1.text(0.5, 0.1, f'Loss Akhir: {best_config["final_loss"]:.6f}', 
         ha='center', va='center', fontsize=12, color='red')

ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.axis('off')

# 2. Performance Comparison with Top 5
top_5 = df.head(5)
config_names = [str(cfg).replace(',', '\n').replace('(', '').replace(')', '') for cfg in top_5['configuration']]

x = np.arange(len(top_5))
width = 0.35

# Create grouped bar chart
ax3.bar(x - width/2, top_5['accuracy'], width, label='Akurasi', color='skyblue', alpha=0.8)
ax3.bar(x + width/2, top_5['final_loss']*10, width, label='Loss x10', color='lightcoral', alpha=0.8)

ax3.set_xlabel('Konfigurasi')
ax3.set_ylabel('Nilai')
ax3.set_title('Perbandingan Top 5 Konfigurasi')
ax3.set_xticks(x)
ax3.set_xticklabels(config_names, rotation=45, ha='right', fontsize=9)
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

# 3. Training Efficiency Analysis
ax2.scatter(df['training_time'], df['accuracy'], alpha=0.6, s=100, c='blue')
ax2.scatter(best_config['training_time'], best_config['accuracy'], 
           s=200, c='red', marker='*', label='Terbaik')
ax2.set_xlabel('Waktu Training (detik)')
ax2.set_ylabel('Akurasi')
ax2.set_title('Efisiensi Training')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 4. Configuration Complexity
df['num_layers'] = df['hidden_layers'].apply(lambda x: len(eval(x)))

ax4.scatter(df['num_layers'], df['accuracy'], alpha=0.6, s=100, c='green')
ax4.scatter(len(eval(best_config['hidden_layers'])), best_config['accuracy'], 
           s=200, c='red', marker='*', label='Terbaik')
ax4.set_xlabel('Jumlah Hidden Layer')
ax4.set_ylabel('Akurasi')
ax4.set_title('Kompleksitas vs Performa')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('output/best_config_summary.png', dpi=300, bbox_inches='tight')
plt.show()

# Create a detailed report
report = f"""
LAPORAN KONFIGURASI TERBAIK MODEL DETEKSI KERIPUT
================================================

1. KONFIGURASI TERBAIK
   - Struktur Hidden Layer: {best_config['configuration']}
   - Akurasi: {best_config['accuracy']:.4f} ({best_config['accuracy']*100:.2f}%)
   - Waktu Training: {best_config['training_time']:.2f} detik
   - Jumlah Iterasi: {best_config['iterations']}
   - Loss Akhir: {best_config['final_loss']:.6f}

2. ANALISIS KINERJA
   - Konfigurasi ini mencapai akurasi tertinggi di antara 40 konfigurasi yang diuji
   - Memiliki waktu training yang relatif cepat dibanding konfigurasi serupa
   - Loss yang sangat rendah menunjukkan konvergensi yang baik

3. REKOMENDASI
   - Gunakan konfigurasi ini untuk deployment model
   - Pertimbangkan early stopping untuk menghemat waktu training
   - Monitor overfitting dengan validasi silang

4. PERBANDINGAN DENGAN KONFIGURASI LAIN
   - Lebih baik {((best_config['accuracy'] - df['accuracy'].mean())/df['accuracy'].std()):.2f} standar deviasi dari rata-rata
   - {((best_config['accuracy'] - df['accuracy'].min())/(df['accuracy'].max() - df['accuracy'].min())*100):.1f}% dari range performa
"""

with open('output/best_config_report.txt', 'w') as f:
    f.write(report)

print("Ringkasan konfigurasi terbaik telah disimpan ke:")
print("- output/best_config_summary.png (visualisasi ringkasan)")
print("- output/best_config_report.txt (laporan lengkap)")
