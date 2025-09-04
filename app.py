from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import cv2
import numpy as np
import mediapipe as mp
import joblib
from werkzeug.utils import secure_filename

# Konfigurasi upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model dan preprocessing tools
model = joblib.load('model/bpnn_model.pkl')
try:
    scaler = joblib.load('model/scaler.pkl')
except:
    scaler = None
try:
    label_encoder = joblib.load('model/label_encoder.pkl')
    if label_encoder is not None:
        print(f"Loaded label encoder classes: {label_encoder.classes_}")  # Debug print
except:
    label_encoder = None

# Inisialisasi face mesh dari MediaPipe
mp_face_mesh = mp.solutions.face_mesh

# Fungsi untuk mengecek ekstensi file
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Fungsi ekstraksi fitur GLCM
def ekstrak_glcm_fitur(area):
    from skimage.feature import graycomatrix, graycoprops
    glcm = graycomatrix(area, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0,0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0,0]
    energy = graycoprops(glcm, 'energy')[0,0]
    return (contrast + homogeneity + energy) / 3

# Fungsi prediksi keriput berdasarkan area wajah
def predict_regions(img):
    results = {}
    img_copy = img.copy()
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)

        # Jika wajah tidak terdeteksi
        if not res.multi_face_landmarks:
            return None, img_copy

        # Landmark wajah
        face_landmarks = res.multi_face_landmarks[0].landmark
        img_h, img_w = img.shape[:2]

        # Koordinat area wajah
        atas = int(face_landmarks[10].y * img_h)
        bawah = int(face_landmarks[152].y * img_h)
        kiri = int(face_landmarks[234].x * img_w)
        kanan = int(face_landmarks[454].x * img_w)
        tinggi = bawah - atas
        h_bagian = tinggi // 4

        # Membuat masker wajah
        face_oval_idx = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        face_oval = np.array([(int(face_landmarks[i].x * img_w), int(face_landmarks[i].y * img_h)) for i in face_oval_idx], np.int32)
        mask_face = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask_face, [face_oval], 255)
        kernel = np.ones((25,25), np.uint8)
        mask_face = cv2.erode(mask_face, kernel, iterations=1)

        # Hilangkan area mata, bibir, hidung agar tidak dihitung
        under_left_eye_idx = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
        under_left_eye = np.array([(int(face_landmarks[i].x * img_w), int(face_landmarks[i].y * img_h)) for i in under_left_eye_idx], np.int32)
        cv2.fillPoly(mask_face, [under_left_eye], 0)

        under_right_eye_idx = [263, 466, 388, 387, 386, 385, 384, 398, 362, 398, 382, 381, 380, 374, 373, 249]
        under_right_eye = np.array([(int(face_landmarks[i].x * img_w), int(face_landmarks[i].y * img_h)) for i in under_right_eye_idx], np.int32)
        cv2.fillPoly(mask_face, [under_right_eye], 0)

        lips_idx = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]
        lips = np.array([(int(face_landmarks[i].x * img_w), int(face_landmarks[i].y * img_h)) for i in lips_idx], np.int32)
        cv2.fillPoly(mask_face, [lips], 0)

        left_eye_idx = [33, 246, 161, 160, 159, 158, 157, 173]
        right_eye_idx = [263, 466, 388, 387, 386, 385, 384, 398]
        left_eye = np.array([(int(face_landmarks[i].x * img_w), int(face_landmarks[i].y * img_h)) for i in left_eye_idx], np.int32)
        right_eye = np.array([(int(face_landmarks[i].x * img_w), int(face_landmarks[i].y * img_h)) for i in right_eye_idx], np.int32)
        cv2.fillPoly(mask_face, [left_eye], 0)
        cv2.fillPoly(mask_face, [right_eye], 0)

        kernel_eye = np.ones((13,13), np.uint8)
        mask_face = cv2.erode(mask_face, kernel_eye, iterations=1)

        left_eyebrow_idx = [70, 63, 105, 66, 107]
        right_eyebrow_idx = [336, 296, 334, 293, 300]
        left_eyebrow = np.array([(int(face_landmarks[i].x * img_w), int(face_landmarks[i].y * img_h)) for i in left_eyebrow_idx], np.int32)
        right_eyebrow = np.array([(int(face_landmarks[i].x * img_w), int(face_landmarks[i].y * img_h)) for i in right_eyebrow_idx], np.int32)
        cv2.fillPoly(mask_face, [left_eyebrow], 0)
        cv2.fillPoly(mask_face, [right_eyebrow], 0)

        nostril_idx = [195, 5, 4, 45, 275, 440, 344, 278, 98, 327, 326, 2, 97, 326, 327, 2]
        nostril = np.array([(int(face_landmarks[i].x * img_w), int(face_landmarks[i].y * img_h)) for i in nostril_idx], np.int32)
        cv2.fillPoly(mask_face, [nostril], 0)
        kernel_nostril = np.ones((15,15), np.uint8)
        mask_face = cv2.erode(mask_face, kernel_nostril, iterations=1)

        # Definisi area wajah untuk analisis
        margin_x = 5
        margin_y = 5
        margin_dahi_top = 10
        area_list = [
            (kiri + margin_x, atas + margin_y + margin_dahi_top, kanan - margin_x, atas + h_bagian - margin_y, 'Dahi'),
            (kiri + margin_x, atas + h_bagian + margin_y, kanan - margin_x, atas + 2*h_bagian - margin_y, 'Mata'),
            (kiri + margin_x, atas + 2*h_bagian + margin_y, kanan - margin_x, atas + 3*h_bagian - margin_y, 'Pipi'),
            (kiri + margin_x, atas + 3*h_bagian + margin_y, kanan - margin_x, bawah - margin_y, 'Mulut')
        ]

        # List untuk menyimpan hasil fitur
        fitur_list = []
        kontur_list = []
        panjang_kontur_list = []

        # Ekstraksi fitur tiap area
        for (x1, y1, x2, y2, label) in area_list:
            roi = img[y1:y2, x1:x2]
            mask_roi = mask_face[y1:y2, x1:x2]

            # Validasi ukuran ROI
            if roi.size == 0 or roi.shape[0] < 10 or roi.shape[1] < 10:
                fitur_list.append(0.0)
                kontur_list.append(0)
                panjang_kontur_list.append(0.0)
                continue

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi

            # Preprocessing berbeda untuk dahi
            if label == 'Dahi':
                clahe_dahi = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(32,32))
                gray_eq_dahi = clahe_dahi.apply(gray)
                adapt = cv2.adaptiveThreshold(gray_eq_dahi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 7)
                sobelx = cv2.Sobel(gray_eq_dahi, cv2.CV_64F, 1, 0, ksize=5)
                sobely = cv2.Sobel(gray_eq_dahi, cv2.CV_64F, 0, 1, ksize=5)
                sobel = cv2.convertScaleAbs(sobelx) + cv2.convertScaleAbs(sobely)
                combined = cv2.add(adapt, sobel)
                canny_low, canny_high = 1, 6
                kernel_close = np.ones((7,7), np.uint8)
            else:
                clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(32,32))
                gray_eq = clahe.apply(gray)
                blur = cv2.GaussianBlur(gray_eq, (3,3), 0)
                combined = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 5)
                canny_low, canny_high = 2, 10
                kernel_close = np.ones((3,3), np.uint8)

            # Ekstrak fitur GLCM
            if label == 'Dahi':
                fitur = ekstrak_glcm_fitur(gray_eq_dahi)
            else:
                fitur = ekstrak_glcm_fitur(gray_eq)
            fitur_list.append(fitur)

            # Deteksi tepi dengan Canny
            edges = cv2.Canny(combined, canny_low, canny_high)
            edges = cv2.bitwise_and(edges, edges, mask=mask_roi)
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close, iterations=1)

            # Thinning agar garis lebih tipis
            try:
                from cv2.ximgproc import thinning
                edges = thinning(edges)
            except Exception:
                pass 

            # Hitung jumlah kontur dan panjangnya
            contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            kontur_list.append(len(contours))
            panjang_total = sum([cv2.arcLength(cnt, False) for cnt in contours])
            panjang_kontur_list.append(panjang_total)

            # Gambar kontur
            for cnt in contours:
                cnt_offset = cnt + np.array([[x1, y1]])
                cv2.drawContours(img_copy, [cnt_offset], -1, (255, 0, 0), 1)

        # Jika fitur tidak lengkap
        if len(fitur_list) != 4:
            for idx, (x1, y1, x2, y2, label) in enumerate(area_list):
                results[label] = {'status': '-', 'nilai': 0.0}
        else:
            # Gabungkan fitur
            jumlah_kontur = sum(kontur_list)
            panjang_total_kontur = sum(panjang_kontur_list)
            fitur_input = fitur_list + [jumlah_kontur, panjang_total_kontur]
            X = np.array([fitur_input])

            # Normalisasi jika scaler ada
            if scaler is not None:
                X = scaler.transform(X)

            # Prediksi model
            try:
                pred = model.predict(X)[0]
                print(f"Prediksi model: {pred}")  # Logging prediksi model

                if label_encoder is not None:
                    try:
                        if pred in label_encoder.classes_:
                            status = label_encoder.inverse_transform([pred])[0]
                            status = status.lower()
                            print(f"Prediksi status: {status}")  # Debug print
                        else:
                            status = str(pred).lower()
                            print(f"Peringatan: Label prediksi tidak dikenal oleh label encoder, menggunakan langsung: {status}")
                    except Exception as e:
                        print(f"Error saat inverse transform label: {e}")
                        status = str(pred).lower()
                else:
                    status = str(pred)

                if not status or status == '-':
                    status = 'tidak diketahui'
            except Exception as e:
                print(f"Error saat prediksi: {e}") 
                status = '-'

            # Evaluasi tiap area
            for idx, (x1, y1, x2, y2, label) in enumerate(area_list):
                nilai_fitur = fitur_list[idx]
                jumlah_kontur_area = kontur_list[idx]
                panjang_kontur_area = panjang_kontur_list[idx]
                area_size = (x2 - x1) * (y2 - y1)
                density_kontur = panjang_kontur_area / max(area_size, 1)
                if nilai_fitur > 20 or jumlah_kontur_area > 12 or density_kontur > 0.008:
                    area_status = 'high'
                else:
                    area_status = 'medium'
                results[label] = {'status': area_status, 'nilai': nilai_fitur}

        return results, img_copy

# Routing halaman utama
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            img = cv2.imread(filepath)
            results, img_with_line = predict_regions(img)
            print(f"DEBUG: results = {results}") 
            output_filename = f"output_{filename}"
            output_path = os.path.join('output', output_filename)
            cv2.imwrite(output_path, img_with_line)
            return render_template('result.html', filename=filename, results=results, output_filename=output_filename)
    return render_template('index.html')

# Routing file upload
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Routing file output
@app.route('/output/<filename>')
def output_file(filename):
    return send_from_directory('output', filename)

# Routing halaman webcam
@app.route('/webcam')
def webcam():
    return render_template('webcam.html')

# Main program
if __name__ == '__main__':
    app.run(debug=True)
