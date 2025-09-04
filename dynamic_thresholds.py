import pandas as pd
import numpy as np

def compute_dynamic_thresholds(csv_path, method='percentile', q=50):
    """
    Hitung threshold dinamis untuk tiap area berdasarkan dataset.
    method: 'mean', 'median', atau 'percentile' (default: median/q=50)
    """
    df = pd.read_csv(csv_path)
    thresholds = {}
    for area in ['dahi', 'mata', 'pipi', 'mulut']:
        if method == 'mean':
            thresholds[area] = df[area].mean()
        elif method == 'percentile':
            thresholds[area] = np.percentile(df[area], q)
        else:
            thresholds[area] = df[area].median()
    return thresholds

if __name__ == '__main__':
    csv_path = 'dataset/keriput_dataset.csv'
    thresholds = compute_dynamic_thresholds(csv_path, method='percentile', q=75)
    print('Dynamic thresholds (75th percentile):')
    for k, v in thresholds.items():
        print(f'{k}: {v:.2f}')
