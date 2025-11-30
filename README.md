**Deteksi Tepi Citra Menggunakan OpenCV – Proyek PCD**

Project ini melakukan preprocessing (grayscale, blur, histogram equalization) dan deteksi tepi (Sobel & Canny) pada dataset gambar menggunakan Python dan OpenCV.

Dataset tidak diupload ke GitHub agar repository tetap ringan. Silakan download dataset melalui link berikut:

Dataset (Google Drive):
👉 https://drive.google.com/drive/folders/1fAUHUyCcsGcwbI1eVmj5B4DMAImzweKA?usp=drive_link

---

## 📁 Struktur Folder Project

```
PCD/
  ├── main.py
  ├── dataset/        # isi gambar (tidak diupload)
  ├── hasil/          # hasil output (tidak diupload)
  ├── .venv/          # environment
  ├── .gitignore
  └── README.md
```

---

## 🛠 Apa Saja yang Perlu Didownload?

1. **Python 3.10**
   [https://www.python.org/downloads/release/python-31011/](https://www.python.org/downloads/release/python-31011/)

2. **Visual Studio Code**
   [https://code.visualstudio.com/](https://code.visualstudio.com/)

3. **Library Python yang diperlukan**
   Jalankan perintah ini setelah mengaktifkan venv:

```
pip install numpy==1.26.4
pip install opencv-python==4.7.0.72
```

---

## 🚀 Cara Menjalankan Project

Ikuti langkah berikut agar orang awam pun bisa mengikuti:

### 1. Clone Repository

```
git clone https://github.com/Reinerbroww/Deteksi_Tepi_PCD.git
cd Deteksi_Tepi_PCD
```

### 2. Buat Virtual Environment

```
python -m venv .venv
```

### 3. Aktifkan Virtual Environment (Windows)

```
.venv\Scripts\activate
```

### 4. Install Semua Library

```
pip install numpy==1.26.4
pip install opencv-python==4.7.0.72
```

### 5. Masukkan Gambar ke Folder dataset/

Masukkan semua file gambar (jpg/png/jpeg) ke folder:

```
dataset/
```

### 6. Jalankan Program

```
python main.py
```

### 7. Lihat Hasil di Folder hasil/

Semua hasil (grayscale, blur, histogram, sobel, canny) otomatis muncul di folder:

```
hasil/
```

---

## 📌 Metode yang Digunakan

* Grayscale
* Gaussian Blur
* Histogram Equalization
* Sobel Edge Detection
* Canny Edge Detection

---

## ✨ Author

Reinnher Sakunab
F55124110 – Teknik Informatika
Universitas Tadulako
