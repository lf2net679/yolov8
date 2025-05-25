import cv2
import time
import numpy as np
from ultralytics import YOLO

# Kamus terjemahan nama objek
TERJEMAHAN = {
    'person': 'SEORANG MANUSIA',
    # 'bicycle': 'Sepeda',
    # 'car': 'Mobil',
    # 'motorcycle': 'Sepeda Motor',
    # 'airplane': 'Pesawat',
    # 'bus': 'Bus',
    # 'train': 'Kereta',
    # 'truck': 'Truk',
    # 'boat': 'Perahu',
    # 'bird': 'Burung',
    # 'cat': 'Kucing',
    # 'dog': 'Anjing',
    # 'horse': 'Kuda',
    # 'sheep': 'Domba',
    # 'cow': 'Sapi',
    # 'chair': 'Kursi',
    # 'couch': 'Sofa',
    # 'bed': 'Tempat Tidur',
    # 'table': 'Meja',
    # 'tv': 'Televisi',
    # 'laptop': 'Laptop',
    # 'mouse': 'Mouse',
    # 'keyboard': 'Keyboard',
    # 'phone': 'Telepon',
    # 'book': 'Buku',
    # 'clock': 'Jam'
}

def terjemahkan_nama(nama):
    """Menerjemahkan nama objek dari bahasa Inggris ke Indonesia"""
    return TERJEMAHAN.get(nama, nama)

def cari_kamera_tersedia():
    """Mencari semua kamera yang tersedia di sistem"""
    kamera_tersedia = []
    for i in range(10):  # Cek indeks 0-9
        cap = cv2.VideoCapture(i)
        
        if cap.isOpened():
            # Baca frame untuk memastikan kamera berfungsi
            ret, frame = cap.read()
            if ret:
                # Dapatkan resolusi kamera
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                kamera_tersedia.append({
                    'index': i,
                    'resolusi': f"{width}x{height}"
                })
            cap.release()
    return kamera_tersedia

def pilih_kamera():
    """Menampilkan menu pilihan kamera dan mengembalikan indeks kamera yang dipilih"""
    kamera_tersedia = cari_kamera_tersedia()
    
    if not kamera_tersedia:
        print("Tidak ada kamera yang terdeteksi!")
        return 0
    
    print("\nKAMERA YANG TERSEDIA:")
    print("=" * 50)
    for kam in kamera_tersedia:
        print(f"[{kam['index']}] Kamera {kam['index']} - Resolusi: {kam['resolusi']}")
    print("=" * 50)
    
    while True:
        try:
            pilihan = int(input("Pilih nomor kamera (0-9): "))
            if any(kam['index'] == pilihan for kam in kamera_tersedia):
                return pilihan
            print("Kamera tidak tersedia, silakan pilih nomor yang valid!")
        except ValueError:
            print("Masukkan nomor yang valid!")

def print_detectable_objects(model):
    """Menampilkan semua objek yang dapat dideteksi oleh model"""
    class_names = model.names
    print("\nOBJEK YANG DAPAT DIDETEKSI:")
    print("=" * 50)
    
    # Menampilkan semua objek dalam format grid
    objects_per_row = 3
    object_list = [terjemahkan_nama(obj) for obj in class_names.values()]
    
    for i in range(0, len(object_list), objects_per_row):
        row = object_list[i:i+objects_per_row]
        print(" | ".join(f"{idx+i}: {obj}" for idx, obj in enumerate(row)))
    
    print("=" * 50)
    print(f"Total objek yang dapat dideteksi: {len(object_list)}")
    print("Tekan 's' untuk keluar dari program")
    print("=" * 50)

def main():
    # Memuat model YOLOv8n (versi ringan)
    model = YOLO('yolov8n.pt')
    
    # Menampilkan semua objek yang dapat dideteksi
    print_detectable_objects(model)
    
    # Memilih kamera yang akan digunakan
    kamera_index = pilih_kamera()
    
    # Menginisialisasi webcam
    cap = cv2.VideoCapture(kamera_index)

    cap.set(cv2.CAP_PROP_FPS, 60)  # Set FPS ke 30
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Set lebar frame ke 320
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Set tinggi frame ke 320
    
    # Memeriksa apakah kamera berhasil dibuka
    if not cap.isOpened():
        print(f"Error: Tidak dapat membuka kamera {kamera_index}")
        return
    
    # Variabel untuk menghitung FPS
    prev_time = 0
    curr_time = 0
    
    while True:
        # Membaca frame dari kamera
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Gagal mengambil frame")
            break

        # Mirror frame duluan
        frame = cv2.flip(frame, 1)
        
        # Menghitung FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
        prev_time = curr_time
        
        # Menjalankan deteksi YOLOv8 pada frame
        results = model(frame, conf=0.5)
        
        # Memproses hasil deteksi
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for i, box in enumerate(boxes):
                # Mendapatkan koordinat kotak pembatas
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Mendapatkan ID kelas dan tingkat kepercayaan
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                # Mendapatkan nama kelas dan terjemahannya
                cls_name = result.names[cls_id]
                label = terjemahkan_nama(cls_name)
                
                # Mengatur warna berdasarkan kelas (hijau untuk manusia, merah untuk lainnya)
                if cls_name == 'person':
                    color = (0, 255, 0)  # Hijau dalam BGR
                else:
                    color = (0, 0, 255)  # Merah dalam BGR
                
                # Menggambar kotak pembatas
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Menampilkan nama kelas dan tingkat kepercayaan
                text = f"{label} {conf:.2f}"
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Menampilkan FPS pada frame
        fps_text = f"FPS: {fps:.1f}"
        # cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Menampilkan frame dengan hasil deteksi
        cv2.imshow('Deteksi Objek', frame)
        
        # Keluar dari loop jika tombol 's' ditekan
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break
    
    # Membersihkan resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
