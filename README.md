# 📊 Klasifikasi Pima Indians Diabetes menggunakan KNN dan Metode Elbow

**Tugas Data Mining - Mahasiswa Informatika**  
**Pure Python Implementation (Tanpa Library Pengolah Data)**

---

## 🎯 Deskripsi Proyek

Proyek ini mengimplementasikan algoritma **K-Nearest Neighbors (KNN)** untuk klasifikasi diabetes pada dataset **Pima Indians Diabetes**. Implementasi dilakukan menggunakan **Python murni** tanpa menggunakan library pengolah data seperti Pandas atau Scikit-learn.

**Metode yang digunakan:**
- ✅ **K-Nearest Neighbors (KNN)** - Algoritma klasifikasi
- ✅ **Elbow Method** - Pencarian nilai k optimal
- ✅ **Min-Max Normalization** - Preprocessing data
- ✅ **Confusion Matrix** - Evaluasi performa
- ✅ **Metrik Evaluasi Lengkap** - Accuracy, Precision, Recall, F1-Score

---

## 📁 Struktur Project

```
Data Mining/
├── main.py                      # Program utama (orchestrator)
├── modules/                     # Package berisi semua modul
│   ├── __init__.py
│   ├── data_loader.py          # Download & load data dari Kaggle
│   ├── preprocessing.py        # Normalisasi & train-test split
│   ├── knn_algorithm.py        # Implementasi KNN dari nol
│   ├── elbow_method.py         # Analisis Elbow untuk k optimal
│   ├── evaluation.py           # Confusion matrix & metrik
│   └── visualization.py        # Grafik Elbow dengan Matplotlib
├── results/                     # Output hasil analisis
│   ├── elbow_analysis.csv      # Hasil analisis untuk semua k
│   ├── confusion_matrix.csv    # Confusion matrix hasil final
│   ├── final_metrics.csv       # Metrik evaluasi (Acc, Prec, Rec, F1)
│   ├── elbow_curve.png         # Grafik Elbow (detailed)
│   └── elbow_curve_simple.png  # Grafik Elbow (simple)
├── requirements.txt            # Dependencies
└── README.md                   # Dokumentasi (file ini)
```

---

## 🚀 Cara Menjalankan

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies:**
- `kagglehub` - Untuk download dataset dari Kaggle
- `matplotlib` - Untuk visualisasi grafik Elbow

### 2. Jalankan Program

```bash
python main.py
```

Program akan otomatis:
1. Download dataset dari Kaggle
2. Preprocessing data (shuffle, split, normalize)
3. Jalankan analisis Elbow (k=1 sampai k=15)
4. Buat grafik visualisasi
5. Train model final dengan k terbaik
6. Evaluasi dan simpan hasil ke CSV

---

## 📊 Output yang Dihasilkan

### 1. **elbow_analysis.csv**
Hasil analisis untuk semua nilai k yang ditest:
```csv
k,error_rate,accuracy
1,35.71,64.29
3,28.57,71.43
5,25.97,74.03
...
```

### 2. **confusion_matrix.csv**
Confusion matrix hasil prediksi final:
```csv
Metric,Value,Description
TP,85,True Positive - Correctly predicted as diabetes
TN,95,True Negative - Correctly predicted as non-diabetes
FP,12,False Positive - Incorrectly predicted as diabetes
FN,8,False Negative - Incorrectly predicted as non-diabetes
```

### 3. **final_metrics.csv**
Metrik evaluasi lengkap:
```csv
Metric,Value,Percentage
Accuracy,0.7662,76.62%
Precision,0.8763,87.63%
Recall,0.9140,91.40%
F1-Score,0.8947,89.47%
```

### 4. **Grafik Elbow**
- `elbow_curve.png` - Grafik lengkap (error rate + accuracy)
- `elbow_curve_simple.png` - Grafik simple (hanya error rate)

---

## 🔬 Metodologi

### 1. **Data Acquisition**
- Dataset: **Pima Indians Diabetes** dari Kaggle
- Sumber: UCI Machine Learning Repository
- Total: 768 samples, 8 features, 1 label
- Features: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
- Label: Outcome (0 = Non-diabetes, 1 = Diabetes)

### 2. **Preprocessing**
**Normalisasi Min-Max:**
```
X_normalized = (X - X_min) / (X_max - X_min)
```
- Rentang: [0, 1]
- Tujuan: Menghindari dominasi fitur dengan nilai besar

**Train-Test Split:**
- Train: 80% (614 samples)
- Test: 20% (154 samples)
- Seed: 42 (untuk reproducibility)

### 3. **KNN Algorithm**
**Euclidean Distance:**
```
distance = √Σ(x1 - x2)²
```

**Proses Klasifikasi:**
1. Hitung jarak dari test sample ke semua training samples
2. Sort jarak (ascending)
3. Ambil k tetangga terdekat
4. Voting mayoritas untuk prediksi label

### 4. **Elbow Method**
**Tujuan:** Mencari nilai k optimal

**Proses:**
1. Test k dari 1 sampai 15
2. Hitung error rate untuk setiap k
3. Plot grafik k vs error rate
4. Pilih k di "siku" grafik (dimana error rate mulai stabil)

**Error Rate:**
```
Error Rate = (Jumlah Prediksi Salah / Total Data) × 100%
```

### 5. **Evaluasi**
**Confusion Matrix:**
```
                Predicted
                0        1
Actual   0    [ TN  |  FP ]
         1    [ FN  |  TP ]
```

**Metrik:**
- **Accuracy** = (TP + TN) / Total
- **Precision** = TP / (TP + FP)
- **Recall (Sensitivity)** = TP / (TP + FN)
- **F1-Score** = 2 × (Precision × Recall) / (Precision + Recall)

---

## 📝 Penjelasan Modul

### 1. `data_loader.py`
- Download dataset dari Kaggle menggunakan `kagglehub`
- Baca CSV menggunakan modul `csv` bawaan Python
- Konversi semua data ke tipe `float`
- Pisahkan features (X) dan labels (y)

### 2. `preprocessing.py`
- Shuffle data dengan seed tetap
- Split data 80:20 (train:test)
- Normalisasi Min-Max (fit pada train, transform pada train & test)

### 3. `knn_algorithm.py`
- Implementasi Euclidean distance manual
- Algoritma pencarian k-nearest neighbors
- Majority voting untuk klasifikasi
- Prediksi untuk batch data

### 4. `elbow_method.py`
- Loop untuk test berbagai nilai k
- Hitung error rate untuk setiap k
- Simpan hasil ke CSV
- Deteksi elbow point otomatis (opsional)

### 5. `evaluation.py`
- Build confusion matrix manual (TP, TN, FP, FN)
- Hitung semua metrik sesuai rumus
- Export hasil ke CSV

### 6. `visualization.py`
- Plot grafik Elbow dengan Matplotlib
- 2 jenis grafik: detailed dan simple
- Annotate best k pada grafik

### 7. `main.py`
- Orchestrator yang menjalankan semua modul
- Error handling lengkap
- Output terstruktur dan informatif

---

## 📈 Interpretasi Hasil

### Grafik Elbow
- **Sumbu X:** Nilai k (1-15)
- **Sumbu Y:** Error rate atau Accuracy
- **Elbow Point:** Titik dimana kurva mulai "mendatar"
- **Best k:** Nilai k di elbow point (trade-off antara bias dan variance)

### Confusion Matrix
- **TP (True Positive):** Model benar prediksi diabetes
- **TN (True Negative):** Model benar prediksi non-diabetes
- **FP (False Positive):** Model salah prediksi diabetes (Type I Error)
- **FN (False Negative):** Model salah prediksi non-diabetes (Type II Error)

### Metrik Evaluasi
- **Accuracy:** Persentase total prediksi yang benar
- **Precision:** Dari semua yang diprediksi diabetes, berapa yang benar
- **Recall:** Dari semua yang sebenarnya diabetes, berapa yang terdeteksi
- **F1-Score:** Harmonic mean dari Precision dan Recall (balance kedua metrik)

---

## ⚠️ Catatan Penting

### Tidak Menggunakan Library Pengolah Data
✅ **Diperbolehkan:**
- `csv` (bawaan Python)
- `math` (bawaan Python)
- `random` (bawaan Python)
- `kagglehub` (hanya untuk download)
- `matplotlib` (hanya untuk visualisasi)

❌ **TIDAK diperbolehkan:**
- `pandas`
- `numpy`
- `scikit-learn`
- Library machine learning lainnya

### Reproducibility
- Semua random operation menggunakan seed=42
- Hasil akan konsisten di setiap run

---

## 🎓 Learning Outcomes

Dari proyek ini, kamu akan memahami:
1. ✅ Cara kerja algoritma KNN dari dasar
2. ✅ Pentingnya normalisasi dalam machine learning
3. ✅ Metode Elbow untuk hyperparameter tuning
4. ✅ Cara menghitung confusion matrix manual
5. ✅ Interpretasi metrik evaluasi (Acc, Prec, Rec, F1)
6. ✅ Implementasi algoritma tanpa library (pure Python)

---

## 📞 Support

Jika ada pertanyaan atau error:
1. Cek file di folder `results/` apakah sudah ter-generate
2. Pastikan semua dependencies ter-install dengan benar
3. Periksa error message di console untuk debugging

---

## 🏆 Kesimpulan

Proyek ini membuktikan bahwa algoritma machine learning dapat diimplementasikan dari nol tanpa library pengolah data. Semua perhitungan (distance, normalization, metrics) dilakukan manual menggunakan Python murni.

**Hasil akhir:**
- Model KNN yang akurat untuk klasifikasi diabetes
- Nilai k optimal yang didapat dari analisis Elbow
- Evaluasi lengkap dengan 4 metrik utama (requirement tugas)
- Visualisasi profesional untuk presentasi

---

**Selamat belajar Data Mining! 🚀**
