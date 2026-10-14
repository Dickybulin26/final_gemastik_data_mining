# 📖 Panduan Lengkap — Gemastik Data Mining

> Panduan ini menjelaskan project klasifikasi teks Bahasa Indonesia untuk orang yang **belum pernah belajar machine learning** sekalipun.

---

## Daftar Isi

1. [Apa Masalah yang Kita Selesaikan?](#1-apa-masalah-yang-kita-selesaikan)
2. [Data yang Kita Punya](#2-data-yang-kita-punya)
3. [Langkah 1 — Membersihkan Teks (Preprocessing)](#3-langkah-1--membersihkan-teks-preprocessing)
4. [Langkah 2 — Mengubah Teks Jadi Angka (TF-IDF)](#4-langkah-2--mengubah-teks-jadi-angka-tf-idf)
5. [Langkah 3 — Melatih Model (Machine Learning)](#5-langkah-3--melatih-model-machine-learning)
6. [Langkah 4 — Menguji & Memilih Model Terbaik](#6-langkah-4--menguji--memilih-model-terbaik)
7. [Langkah 5 — Membuat Prediksi Akhir](#7-langkah-5--membuat-prediksi-akhir)
8. [Hasil Akhir](#8-hasil-akhir)
9. [Penjelasan Istilah Teknis](#9-penjelasan-istilah-teknis)
10. [Cara Menjalankan Kode](#10-cara-menjalankan-kode)

---

## 1. Apa Masalah yang Kita Selesaikan?

Di internet, banyak komentar dan teks yang mengandung ujaran kebencian, kata-kata kasar, atau konten negatif. Memeriksa satu per satu oleh manusia tentu tidak efisien. 

**Solusi:** Kita membangun program komputer yang bisa **otomatis mendeteksi** apakah sebuah teks bersifat **toxic (kasar/negatif)** atau **positif/netral**.

### Contoh:
| Teks | Label | Artinya |
|------|-------|---------|
| "Terima kasih sudah hadir, semoga bermanfaat" | 0 | ✅ Positif |
| "Anjing emang lu, capek dengerin bullshit lo" | 1 | ❌ Toxic |
| "Ayo semangat! Pasti bisa! 💪" | 0 | ✅ Positif |
| "Dasar gak guna, nyia-nyiain kesempatan" | 1 | ❌ Toxic |

---

## 2. Data yang Kita Punya

Kompetisi menyediakan 3 file data:

### a) `labeled.csv` — Data Latih (800 baris)
Ini adalah data yang **sudah ada jawabannya** (label 0 atau 1). Kita gunakan ini untuk "mengajari" komputer.

### b) `unlabeled.csv` — Data Tambahan (2.400 baris)  
Teks tanpa label. Kita **tidak menggunakannya untuk melatih**, tapi untuk memperkaya kosakata. Ibarat memberi komputer "kamus" yang lebih lengkap.

### c) `test.csv` — Data Ujian (400 baris)
Teks yang **harus kita prediksi** labelnya. Inilah "soal ujian" dari kompetisi.

---

## 3. Langkah 1 — Membersihkan Teks (Preprocessing)

Teks dari internet itu "kotor" — ada URL, mention (@user), hashtag, angka, dan slang. Komputer akan bingung kalau langsung diproses. Maka kita bersihkan dulu:

### Yang kita lakukan:

| Proses | Sebelum | Sesudah |
|--------|---------|---------|
| Huruf kecil semua | "KAMU Bego!" | "kamu bego!" |
| Hapus URL | "cek http://t.co/abc" | "cek" |
| Hapus mention | "@admin tolong" | "tolong" |
| Hapus hashtag | "#gemastik bagus" | "bagus" |
| Hapus angka | "ada 100 orang" | "ada orang" |
| Hapus tanda baca | "halo!!!" | "halo" |
| Normalisasi slang | "gw gak mau" | "saya tidak mau" |
| Squash huruf berulang | "baguuuus" | "bagus" |

### Kenapa normalisasi slang penting?
Orang Indonesia sering menulis dengan slang: "gw", "gak", "emg", "bgt", dll. Tanpa normalisasi, komputer menganggap "tidak" dan "gak" sebagai kata yang **berbeda**. Dengan mengganti slang ke bentuk baku, komputer bisa memahami bahwa keduanya bermakna sama.

Kami membuat kamus **70+ kata slang** yang umum dipakai di internet Indonesia.

---

## 4. Langkah 2 — Mengubah Teks Jadi Angka (TF-IDF)

Komputer tidak bisa "membaca" teks — yang bisa diproses komputer hanyalah angka. Maka kita perlu cara untuk **mengubah teks menjadi representasi angka**.

### Apa itu TF-IDF?

**TF-IDF** (*Term Frequency - Inverse Document Frequency*) adalah cara mengukur **seberapa penting sebuah kata** dalam suatu dokumen.

**Analoginya:**
- Bayangkan kamu membaca 100 artikel tentang berbagai topik
- Kata "dan", "yang", "di" muncul di **semua** artikel → kata ini **tidak penting** (nilai rendah)
- Kata "toxic" hanya muncul di **beberapa** artikel tertentu → kata ini **penting** untuk membedakan artikel tersebut (nilai tinggi)

### Pengaturan TF-IDF kami:
- **50.000 fitur** — kita ambil 50.000 kata/frasa terpenting
- **Unigram + Bigram** — kita lihat kata tunggal ("tidak") DAN pasangan kata ("tidak mau")
- **Transductive** — kita fit TF-IDF pada gabungan SEMUA teks (labeled + unlabeled + test) agar kosakata lebih lengkap. Ini **bukan** kebocoran data karena kita hanya menghitung frekuensi kata, bukan menggunakan label.

---

## 5. Langkah 3 — Melatih Model (Machine Learning)

Setelah teks menjadi angka, kita "mengajarkan" komputer pola-pola yang membedakan teks toxic dan non-toxic. Kami menguji **3 model**:

### a) Multinomial Naive Bayes (NB)
- **Analogi:** Seperti menghitung peluang. "Kalau sebuah teks mengandung kata 'anjing' dan 'bego', berapa peluang teks itu toxic?"
- **Kelebihan:** Sangat cepat, sederhana
- **Kekurangan:** Menganggap setiap kata independen (tidak memperhatikan konteks)

### b) Logistic Regression (LR)
- **Analogi:** Menarik garis batas di antara data. Teks di sisi kiri = positif, di sisi kanan = toxic.
- **Kelebihan:** Lebih fleksibel dari NB, bisa memberi probabilitas
- **Kekurangan:** Butuh tuning parameter

### c) Linear SVC (Support Vector Classifier) ⭐
- **Analogi:** Sama seperti LR (menarik garis batas), tapi mencari garis yang **paling jauh** dari kedua kelompok. Semakin jauh garis dari data terdekat, semakin yakin model dalam keputusannya.
- **Kelebihan:** Sangat efektif untuk data teks, terbukti di banyak riset
- **Kekurangan:** Tidak langsung memberikan probabilitas

---

## 6. Langkah 4 — Menguji & Memilih Model Terbaik

### Apa itu Cross-Validation?

Kita tidak bisa langsung menguji model pada data test (karena kita tidak tahu jawabannya). Solusinya: **5-Fold Stratified Cross-Validation**.

**Cara kerjanya:**
1. Bagi 800 data latih menjadi **5 bagian** (fold) yang sama rata
2. Latih model pada 4 bagian, uji pada 1 bagian sisanya
3. Ulangi 5 kali (setiap bagian pernah jadi "soal ujian")
4. Rata-ratakan hasilnya

Ini seperti ujian yang **diadakan 5 kali dengan soal berbeda** — hasilnya lebih terpercaya daripada ujian sekali saja.

### Apa itu Macro F1-Score?

F1-Score mengukur keseimbangan antara:
- **Precision** — Dari yang kita tebak "toxic", berapa yang benar-benar toxic?
- **Recall** — Dari semua yang memang toxic, berapa yang berhasil kita temukan?

**Macro** artinya kita hitung F1 untuk setiap kelas (0 dan 1), lalu **rata-ratakan**. Ini adil untuk kedua kelas.

**Skala:** 0 (sangat buruk) sampai 1 (sempurna).

### Hasil:
| Model | F1-Score | Keterangan |
|-------|----------|------------|
| Naive Bayes | 0.8486 | Baseline yang bagus |
| Logistic Regression | 0.8674 | Lebih baik dari NB |
| **Linear SVC** | **0.8734** | **Terbaik!** ⭐ |

---

## 7. Langkah 5 — Membuat Prediksi Akhir

Setelah mengetahui Linear SVC adalah model terbaik:

1. **Latih ulang** Linear SVC pada **semua 800 data** (sebelumnya hanya 640 per fold)
2. **Prediksi** 400 teks di test.csv
3. **Simpan** hasilnya sebagai `submission.csv`
4. **Validasi** format dengan `check_submission.py`

---

## 8. Hasil Akhir

✅ Model **Linear SVC** berhasil mendeteksi teks toxic dengan **akurasi F1 ≈ 0.87**.

### Kenapa hasilnya bagus?
1. **Normalisasi slang** — membuat kata-kata informal bisa dikenali
2. **TF-IDF transductive** — kosakata lebih kaya karena fit pada semua teks
3. **Linear SVC** — model yang tepat untuk fitur sparse berdimensi tinggi

### Potensi perbaikan di masa depan:
1. **Semi-supervised learning** — memanfaatkan 2.400 data unlabeled untuk training
2. **Word embeddings** (FastText/Word2Vec) — representasi kata yang lebih kaya
3. **IndoBERT** — model deep learning khusus Bahasa Indonesia
4. **Ensemble** — menggabungkan beberapa model untuk akurasi lebih tinggi

---

## 9. Penjelasan Istilah Teknis

| Istilah | Penjelasan Sederhana |
|---------|---------------------|
| **Machine Learning** | Komputer belajar dari data, bukan diprogram secara manual |
| **Klasifikasi** | Mengelompokkan data ke dalam kategori (misal: toxic vs non-toxic) |
| **Preprocessing** | Membersihkan dan menyiapkan data sebelum diproses |
| **TF-IDF** | Cara mengubah teks jadi angka berdasarkan seberapa penting sebuah kata |
| **Feature** | Ciri-ciri yang digunakan model untuk membuat keputusan |
| **Training** | Proses "mengajari" model dari data yang sudah ada jawabannya |
| **Cross-Validation** | Menguji model berkali-kali dengan pembagian data berbeda |
| **F1-Score** | Metrik yang mengukur akurasi secara seimbang (0–1) |
| **Overfitting** | Model terlalu "hafal" data latih sehingga buruk pada data baru |
| **SVC/SVM** | Support Vector Machine — model yang mencari batas pemisah terbaik |
| **Naive Bayes** | Model berdasarkan teori probabilitas |
| **Transductive** | Memanfaatkan data test (tanpa label) saat menyiapkan fitur |
| **Sparse** | Data yang sebagian besar nilainya nol (umum di TF-IDF) |

---

## 10. Cara Menjalankan Kode

### Prasyarat
- Python 3.8 atau lebih baru
- pip (package manager Python)

### Langkah-langkah

```bash
# 1. Install library
pip install -r requirements.txt

# 2. Jalankan analisis data (opsional, grafik sudah ada di figures/)
python -m src.eda.run_eda

# 3. Jalankan eksperimen & buat prediksi
python -m src.modeling.run_experiments

# 4. Cek format submission
python check_submission.py
```

### Urutan File yang Dieksekusi

```
run_eda.py
  └── utils.py (fungsi bantu EDA)
       ├── Hitung statistik dasar
       ├── Buat grafik distribusi kelas
       ├── Buat grafik panjang teks
       ├── Buat grafik top bigrams
       └── Analisis OOV (Out-of-Vocabulary)

run_experiments.py
  ├── preprocessing.py (bersihkan teks)
  ├── features.py (TF-IDF)
  ├── Latih 3 model + cross-validation
  ├── Pilih model terbaik
  └── Buat submission.csv
```
