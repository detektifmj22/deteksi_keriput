import cv2
import numpy as np
import mediapipe as mp
import joblib

# Load label encoder jika tersedia
try:
    label_encoder = joblib.load('model/label_encoder.pkl')
except:
    label_encoder = None

# Load model utama
model = joblib.load('model/bpnn_model.pkl')

# Load scaler jika tersedia
try:
    scaler = joblib.load('model/scaler.pkl')
except:
    scaler = None

# Akses kamera (webcam)
cap = cv2.VideoCapture(0)

# Inisialisasi modul FaceMesh dari MediaPipe
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Fungsi ekstraksi fitur GLCM
def ekstrak_glcm_fitur(area):
    import skimage.feature
    glcm = skimage.feature.graycomatrix(area, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = skimage.feature.graycoprops(glcm, 'contrast')[0,0]
    homogeneity = skimage.feature.graycoprops(glcm, 'homogeneity')[0,0]
    energy = skimage.feature.graycoprops(glcm, 'energy')[0,0]
    return (contrast + homogeneity + energy) / 3

# Fungsi menggambar area wajah + ekstraksi fitur
def draw_face_areas_landmarks(img, landmarks, img_w, img_h):
    
    # Tentukan batas area wajah berdasarkan landmark
    atas = int(landmarks[10].y * img_h)
    bawah = int(landmarks[152].y * img_h)
    kiri = int(landmarks[234].x * img_w)
    kanan = int(landmarks[454].x * img_w)
    tinggi = bawah - atas
    h_bagian = tinggi // 4

    # Daftar area wajah (dahi, mata, pipi, mulut)
    area_list = [
        (kiri, atas, kanan, atas + h_bagian, (0,255,255), 'Dahi'),
        (kiri, atas + h_bagian, kanan, atas + 2*h_bagian, (0,255,0), 'Mata'),
        (kiri, atas + 2*h_bagian, kanan, atas + 3*h_bagian, (255,0,255), 'Pipi'),
        (kiri, atas + 3*h_bagian, kanan, bawah, (0,0,255), 'Mulut')
    ]
    fitur_list = []
    import cv2
    
    # Fungsi preprocessing ROI (CLAHE + grayscale)
    def preprocess_roi(roi):
        if roi.size == 0 or roi.shape[0] < 10 or roi.shape[1] < 10:
            return None
        roi_resized = cv2.resize(roi, (200, 200))
        gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY) if len(roi_resized.shape) == 3 else roi_resized
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray_clahe = clahe.apply(gray)
        return gray_clahe

    # Looping untuk setiap area (dahi, mata, pipi, mulut)
    for (x1, y1, x2, y2, color, label) in area_list:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)  # Gambarkan kotak
        roi = img[y1:y2, x1:x2]  # Potong ROI
        if roi.size == 0 or roi.shape[0] < 10 or roi.shape[1] < 10:
            print(f"ROI kosong untuk area {label}")
            fitur_list.append(None)
            continue
        preprocessed = preprocess_roi(roi)
        if preprocessed is None:
            print(f"Preprocessing gagal untuk area {label}")
            fitur_list.append(None)
            continue
        fitur = ekstrak_glcm_fitur(preprocessed)
        print(f"Nilai fitur {label}: {fitur}")
        fitur_list.append(fitur)

    # Hitung kontur (jumlah & panjang total)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)
    edges = cv2.Canny(img_blur, 30, 80)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    jumlah_kontur = len(contours)
    panjang_total_kontur = sum([cv2.arcLength(cnt, False) for cnt in contours])
    fitur_list.append(jumlah_kontur)
    fitur_list.append(panjang_total_kontur)

    # Prediksi dan tampilkan status di gambar
    for idx, (x1, y1, x2, y2, color, label) in enumerate(area_list):
        fitur = fitur_list[idx]
        if None in fitur_list:  # Jika ada ROI gagal diproses
            status = '-'
            nilai = 0.0
        else:
            nilai = fitur
            X_pred = np.array([fitur_list])
            if scaler is not None:
                X_pred = scaler.transform(X_pred)
            pred = model.predict(X_pred)
         
            if label_encoder is not None:
                status = label_encoder.inverse_transform(pred)[0]
            else:
                status = str(pred[0])
            
            # Format label prediksi agar lebih rapi
            status = status.replace('_', ' ')
            status = ' '.join([w.capitalize() for w in status.split()])
            print(f"Prediksi area {label}: {status} (fitur: {nilai})")
            
        # Tampilkan hasil prediksi di layar
        text = f"{label}: {status} ({nilai:.2f})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.1
        thickness = 3
        
        cv2.putText(img, text, (x1+8, y1+34), font, font_scale, (0,0,0), thickness+2, cv2.LINE_AA)
        cv2.putText(img, text, (x1+5, y1+30), font, font_scale, (0,255,255), thickness, cv2.LINE_AA)

# Jalankan FaceMesh dan looping kamera
with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
       
        img_h, img_w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                draw_face_areas_landmarks(frame, face_landmarks.landmark, img_w, img_h)
        cv2.imshow('Deteksi Keriput (Tekan q untuk keluar)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
