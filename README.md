# 🛒 E-Commerce Customer Churn Prediction

Proyek machine learning untuk memprediksi pelanggan yang akan berhenti berlangganan (*churn*) pada platform e-commerce, menggunakan pendekatan end-to-end dari data cleaning hingga evaluasi model.

---

## 📋 Deskripsi Proyek

Churn pelanggan adalah salah satu tantangan terbesar bisnis e-commerce. Biaya mendapatkan pelanggan baru jauh lebih mahal dibanding mempertahankan yang sudah ada. Proyek ini membangun model klasifikasi biner untuk mengidentifikasi pelanggan berisiko churn **sebelum** mereka benar-benar pergi, sehingga tim bisnis dapat melakukan intervensi tepat waktu.

---

## 📁 Struktur Repository

```
├── Capstone_2.ipynb                    # Notebook utama (end-to-end pipeline)
├── data_ecommerce_customer_churn.csv   # Dataset
└── README.md
```

---

## 📊 Dataset

| Atribut | Detail |
|---------|--------|
| **Sumber** | E-Commerce Customer Churn Dataset |
| **Jumlah baris** | 3.941 pelanggan |
| **Jumlah kolom** | 11 kolom |
| **Target** | `Churn` (0 = Tidak Churn, 1 = Churn) |
| **Class imbalance** | ~83% tidak churn vs ~17% churn |

### Fitur Dataset

| Kolom | Tipe | Deskripsi |
|-------|------|-----------|
| `Tenure` | Numerik | Lama berlangganan (bulan) |
| `WarehouseToHome` | Numerik | Jarak gudang ke rumah pelanggan |
| `NumberOfDeviceRegistered` | Numerik | Jumlah perangkat terdaftar |
| `PreferedOrderCat` | Kategorikal | Kategori produk yang paling sering dipesan |
| `SatisfactionScore` | Numerik | Skor kepuasan pelanggan (1–5) |
| `MaritalStatus` | Kategorikal | Status pernikahan |
| `NumberOfAddress` | Numerik | Jumlah alamat yang tersimpan |
| `Complain` | Biner | Pernah mengajukan komplain (0/1) |
| `DaySinceLastOrder` | Numerik | Hari sejak pesanan terakhir |
| `CashbackAmount` | Numerik | Total cashback yang diterima |
| `Churn` | Biner | **Target** — pelanggan churn atau tidak |

---

## ⚙️ Pipeline

### Bagian 1 — Data Cleaning
- **Missing Values** — deteksi dan imputasi: median untuk kolom numerik, modus untuk kolom kategorikal
- **Duplicate Data** — deteksi dan penghapusan baris duplikat
- **Inconsistent Data** — standardisasi teks (strip, title case), penggabungan kategori mirip (`Mobile` → `Mobile Phone`)
- **Outlier Handling** — deteksi dengan metode IQR, penanganan dengan *Winsorization* (capping)

### Bagian 2 — Data Preprocessing
- **Encoding** — One-Hot Encoding untuk kolom kategorikal nominal (`PreferedOrderCat`, `MaritalStatus`)
- **Splitting** — Train 80% / Test 20% dengan `stratify=y` untuk menjaga proporsi kelas
- **Scaling** — StandardScaler pada kolom numerik (fit hanya pada training data untuk mencegah data leakage)
- **Balancing** — SMOTE (*Synthetic Minority Over-sampling Technique*) untuk menangani class imbalance

### Bagian 3 — Hyperparameter Tuning
- **Baseline Comparison** — perbandingan 4 model: Logistic Regression, Random Forest, Gradient Boosting, XGBoost menggunakan Stratified K-Fold CV (5-fold)
- **Tuning** — `RandomizedSearchCV` (30 iterasi) pada Random Forest dan XGBoost
- **Retrain** — model dilatih ulang dengan parameter terbaik pada seluruh training data (post-SMOTE)
- **CV Validation** — cross-validation ulang untuk memvalidasi stabilitas model tuned

### Bagian 4 — Final Evaluation
- Evaluasi pada test set yang tidak pernah dilihat model
- Classification Report per kelas
- Confusion Matrix (fokus pada False Negative)
- ROC Curve & AUC Score
- Feature Importance (Top 15 fitur)

---

## 🤖 Model

| Model | Keterangan |
|-------|-----------|
| Logistic Regression | Baseline linear |
| Random Forest | Ensemble bagging — baseline & tuned |
| Gradient Boosting | Baseline boosting |
| **XGBoost** | Gradient boosting — baseline & **tuned (model utama)** |

---

## 📈 Metrik Evaluasi

Karena dataset imbalanced, akurasi saja tidak cukup. Metrik yang digunakan:

| Metrik | Alasan |
|--------|--------|
| **Recall** | Metrik utama — meminimalkan *missed churn* (False Negative) |
| **Precision** | Menghindari intervensi sia-sia ke non-churn |
| **F1 Score** | Keseimbangan antara Precision dan Recall |
| **ROC-AUC** | Kemampuan diskriminasi model secara keseluruhan |
| **Accuracy** | Informasi pelengkap |

---

## 🛠️ Instalasi & Cara Menjalankan

### 1. Clone Repository
```bash
git clone https://github.com/username/capstone-2-churn-prediction.git
cd capstone-2-churn-prediction
```

### 2. Install Dependencies
```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost
```

### 3. Jalankan Notebook
```bash
jupyter notebook Capstone_2.ipynb
```

### Atau jalankan di Google Colab
1. Upload `Capstone_2.ipynb` ke Google Colab
2. Upload `data_ecommerce_customer_churn.csv` saat diminta
3. Jalankan semua cell secara berurutan

---

## 📦 Dependencies

```
numpy
pandas
matplotlib
seaborn
scikit-learn
imbalanced-learn
xgboost
```

---

## 🔑 Key Findings

- **SMOTE** secara signifikan meningkatkan Recall model terhadap kelas churn
- **XGBoost** dengan hyperparameter tuning menghasilkan ROC-AUC tertinggi
- Fitur paling berpengaruh terhadap churn antara lain: `Tenure`, `Complain`, `SatisfactionScore`, dan `DaySinceLastOrder`
- **False Negative** (churn tidak terdeteksi) adalah kesalahan yang paling mahal secara bisnis — model dioptimalkan untuk meminimalkannya

---

## 👤 Author

**Capstone Project 2** — Machine Learning Pipeline untuk E-Commerce Churn Prediction
