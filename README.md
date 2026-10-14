# 🇮🇩 Klasifikasi Teks Bahasa Indonesia — Gemastik Data Mining

## 📌 Tentang Project Ini

Project ini bertujuan untuk **mendeteksi teks toxic (kasar/negatif) dalam Bahasa Indonesia** secara otomatis menggunakan teknik *machine learning*. Sistem ini membaca sebuah teks, lalu menentukan apakah teks tersebut:

- **Label 0** → Teks **positif / netral** (tidak toxic)
- **Label 1** → Teks **toxic / negatif** (mengandung ujaran kasar)

> 💡 **Analogi sederhana:** Bayangkan sebuah filter komentar di media sosial yang bisa otomatis menandai komentar kasar — itulah yang dilakukan project ini!

---

## 🏆 Hasil

Model terbaik kami (**Linear SVC**) mencapai akurasi **F1-Score = 0.87** (dari skala 0–1), yang artinya sistem ini benar **~87% dari waktu** dalam mendeteksi teks toxic maupun non-toxic.

| Model | Skor F1 | Keterangan |
|-------|---------|------------|
| Multinomial Naive Bayes | 0.8486 | Baseline cepat |
| Logistic Regression | 0.8674 | Model linear standar |
| **Linear SVC** ⭐ | **0.8734** | **Model terbaik** |

---

## 📂 Struktur Project

```
📦 final_gemastik_data_mining/
│
├── 📊 data/                    ← Data mentah (CSV)
│   ├── labeled.csv             ← 800 teks + label untuk training
│   ├── unlabeled.csv           ← 2.400 teks tanpa label (tambahan kosakata)
│   ├── test.csv                ← 400 teks yang harus diprediksi
│   └── submission_template.csv ← Format jawaban yang diminta
│
├── 🔬 src/                     ← Kode utama
│   ├── eda/                    ← Analisis data (EDA)
│   │   ├── utils.py            ← Fungsi-fungsi bantu untuk EDA
│   │   └── run_eda.py          ← Script menjalankan seluruh EDA
│   ├── modeling/               ← Pipeline machine learning
│   │   ├── preprocessing.py    ← Pembersihan & normalisasi teks
│   │   ├── features.py         ← Ekstraksi fitur (TF-IDF)
│   │   └── run_experiments.py  ← Eksperimen model & prediksi
│   └── legacy/                 ← Arsip script lama (referensi)
│
├── 📈 figures/                 ← Grafik hasil analisis (auto-generated)
├── 📋 results/                 ← Ringkasan hasil eksperimen
├── 📖 docs/                    ← Dokumentasi lengkap
│   └── PANDUAN.md              ← 👈 Panduan lengkap untuk pemula
│
├── 🛠️ tools/                   ← Script pembantu
│   ├── create_notebook.py      ← Generator notebook Jupyter
│   └── generate_presentation_pptx.py ← Generator slide presentasi
│
├── 📓 notebook.ipynb           ← Notebook Jupyter (deliverable)
├── 🎤 presentation.pptx       ← Slide presentasi (deliverable)
├── 📤 submission.csv           ← Prediksi akhir untuk kompetisi
├── 📋 requirements.txt        ← Daftar library yang dibutuhkan
└── ✅ check_submission.py      ← Validasi format submission
```

---

## 🚀 Cara Menjalankan

### Persiapan (satu kali)
```bash
# 1. Install semua library yang dibutuhkan
pip install -r requirements.txt
```

### Menjalankan Pipeline
```bash
# 2. Jalankan analisis data (EDA) → menghasilkan grafik di figures/
python -m src.eda.run_eda

# 3. Jalankan eksperimen model → menghasilkan submission.csv
python -m src.modeling.run_experiments

# 4. Cek apakah submission.csv sudah benar formatnya
python check_submission.py
```

### Opsional
```bash
# Buat ulang notebook Jupyter
python tools/create_notebook.py

# Buat ulang slide presentasi
python tools/generate_presentation_pptx.py
```

---

## 🧠 Bagaimana Sistem Ini Bekerja?

### Langkah 1: Bersihkan Teks
Teks mentah dari internet penuh dengan noise (URL, emoji, slang). Kami membersihkannya:
- `"Kamu tuh udah dikasih kesempatan, malah nyia-nyiain!"` 
- → `"kamu itu sudah dikasih kesempatan malah nyia nyiain"`

### Langkah 2: Ubah Teks Jadi Angka (TF-IDF)
Komputer tidak bisa membaca teks — harus diubah jadi angka. **TF-IDF** menghitung seberapa penting setiap kata dalam sebuah dokumen:
- Kata yang **sering muncul di satu teks tapi jarang di teks lain** → nilai tinggi (penting!)
- Kata yang **muncul di mana-mana** (seperti "yang", "dan") → nilai rendah

### Langkah 3: Latih Model Machine Learning
Kami menguji 3 model dan memilih yang terbaik:
- **Naive Bayes** — model paling sederhana, menghitung probabilitas
- **Logistic Regression** — model yang menarik garis batas pemisah
- **Linear SVC** ⭐ — model terbaik, mencari garis pemisah yang paling optimal

### Langkah 4: Prediksi
Model terbaik dilatih ulang dengan **semua data training**, lalu memprediksi 400 teks di test set.

---

## 📚 Dokumentasi Lebih Lanjut

- [Panduan Lengkap untuk Pemula](docs/PANDUAN.md) — penjelasan detail setiap konsep
- [Brief Kompetisi](docs/BRIEF.md) — deskripsi tugas kompetisi
- [Lembar Persiapan Presentasi](docs/defense_sheet.md) — Q&A untuk sidang

---

## 👥 Tim

> *(Isi nama tim dan anggota di sini)*

**Kompetisi:** Gemastik – Data Mining