import pandas as pd
import matplotlib.pyplot as plt

csv_path = 'dataset/keriput_dataset.csv'
df = pd.read_csv(csv_path)

areas = ['dahi', 'mata', 'pipi', 'mulut']
plt.figure(figsize=(12,8))
for i, area in enumerate(areas, 1):
    plt.subplot(2,2,i)
    df[area].hist(bins=40)
    plt.title(f'Distribusi Nilai Fitur: {area}')
    plt.xlabel('Nilai Fitur')
    plt.ylabel('Frekuensi')
plt.tight_layout()
plt.show()
