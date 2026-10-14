# 03. Panduan Deliverables: Notebook, 5 Slide, & Pertahanan Juri

> **Petunjuk Pelaksana:** Dokumen ini memuat spesifikasi persis susunan `notebook.ipynb`, naskah 5 kalimat kesimpulan, draf isi 5 slide presentasi (5 menit), serta skrip pertahanan jawaban untuk 3 pertanyaan dewan juri (5 menit).

---

## 1. Blueprint Struktur Notebook (`notebook.ipynb`)

Sesuai aturan kompetisi, seluruh alur kerja harus terangkum dalam **satu notebook mandiri** (*single reproducible notebook*) yang dapat dieksekusi secara mulus dari atas ke bawah.

### Susunan Sel Notebook:

1. **Sel Markdown 1: Judul & Latar Belakang SDG 8 & 9**
   * *Judul:* "Semi-Supervised Sentiment Classification: Mengoptimalkan Nilai Bisnis Ulasan Konsumen dengan Anggaran Anotasi Terbatas".
   * *Naratif:* Masalah anggaran 300 label, pemanfaatan 10.633 data ulasan mentah, relevansi SDG 8 (efisiensi kerja/SDM anotasi) dan SDG 9 (infrastruktur AI adaptif).
2. **Sel Kode 1 & Markdown 2: Pemuatan Data & Eksplorasi (EDA)**
   * Membaca dataset, memeriksa missing value.
   * Visualisasi: Grafik batang distribusi kelas (bukti ketimpangan tajam: 58.3% positif, 31.3% negatif, 10.3% netral).
   * Visualisasi: Histogram panjang kata (bukti kesamaan distribusi ~33 kata lintas dataset).
3. **Sel Kode 2 & Markdown 3: Bukti Kuantitatif Jurang OOV (Out-of-Vocabulary)**
   * Menghitung dan memvisualisasikan OOV Rate pada data uji: 70.49% (hanya dengan 300 label) anjlok menjadi 18.15% (dengan 10.633 data tanpa label).
   * Penjelasan ilmiah mengapa data tanpa label krusial untuk ekspansi leksikal.
4. **Sel Kode 3 & Markdown 4: Skema Validasi Stratified 5-Fold & Preprocessing**
   * Preprocessing teks cerdas (`clean_text`).
   * Pembangunan `FeatureUnion` (Word n-gram (1,2) + Char n-gram (3,5)).
   * Transductive vocabulary fitting pada seluruh korpus.
5. **Sel Kode 4 & Markdown 5: Benchmark Model Baseline Supervised (300 Label)**
   * Evaluasi Logistic Regression, LinearSVC, ComplementNB, dan Voting Classifier pada 5-Fold CV.
   * Cetak tabel skor Macro-F1 per fold dan rata-rata.
6. **Sel Kode 5 & Markdown 6: Implementasi Semi-Supervised Learning (CB-PL)**
   * Eksekusi Class-Balanced Pseudo-Labeling dengan sample weighting.
   * Eksperimen fold-by-fold bebas kebocoran data.
7. **Sel Kode 6 & Markdown 7: Tabel Komparasi Kuantitatif & Diagnostik Pseudo-Label**
   * Tabel komparasi: Baseline vs Semi-Supervised (Macro-F1).
   * Grafik diagnostik: Jumlah sampel pseudo-label yang diserap per kelas, rata-rata tingkat keyakinan (*confidence*), dan analisis apakah terjadi *confirmation bias*.
8. **Sel Kode 7: Inferensi Akhir & Verifikasi File Submission**
   * Fit model final pada data latih penuh + pseudo-label terpilih.
   * Ekspor `submission.csv` dan jalankan script verifikasi `check_submission.py`.
9. **Sel Markdown Akhir: Kesimpulan Akhir (Maksimal 5 Kalimat)**
   * Kesimpulan formal, padat, dan ilmiah.

---

## 2. Naskah Kesimpulan Akhir Notebook (Maksimal 5 Kalimat)

> *"Pemanfaatan 10.633 data tanpa label terbukti secara kuantitatif memangkas angka Out-of-Vocabulary (OOV) data uji dari 70,49% menjadi 18,15%, memperluas cakupan leksikal model secara masif. Melalui metode Class-Balanced Pseudo-Labeling, kami berhasil menambahkan pseudo-label berkualitas tinggi tanpa memicu confirmation bias pada kelas mayoritas positif. Hasil validasi Stratified 5-Fold CV menunjukkan peningkatan Macro-F1 yang konsisten dibandingkan baseline supervised murni 300 data. Pembobotan sampel terkalibrasi efektif mencegah perambatan derau (noise propagation) dari data ulasan ambigu. Dengan demikian, data tanpa label terbukti sangat membantu performa model dan menjadi solusi hemat biaya yang selaras dengan efisiensi industri (SDG 8 & 9)."*

---

## 3. Materi 5 Slide Presentasi (Durasi: 5 Menit)

Gunakan struktur slide ringkas dan berdampak tinggi berikut untuk presentasi final:

### Slide 1: Masalah Bisnis & Keselarasan Tema SDG
* **Judul:** Inovasi Klasifikasi Sentimen Hemat Biaya Berbasis Semi-Supervised Learning.
* **Latar Belakang:** Anggaran industri terbatas—hanya mampu melabeli 300 dari belasan ribu ulasan konsumen.
* **Pertanyaan Inti:** Apakah data tanpa label merupakan aset bernilai (*boon*) atau justru racun (*bane*) bagi model?
* **Dampak & SDG:**
  * **SDG 8 (Decent Work):** Efisiensi biaya operasional dan optimasi beban kerja tim anotasi manusia.
  * **SDG 9 (Industry & Innovation):** Pemanfaatan data mentah tak berstruktur skala besar menjadi inteligensi bisnis adaptif.

### Slide 2: Temuan Kritis Eksplorasi Data (EDA)
* **Temuan 1: Ketimpangan Kelas Ekstrem (Severe Imbalance):**
  * Positif 58.3%, Negatif 31.3%, Netral hanya 10.3%.
  * Risiko: Model naif akan mengalami *Confirmation Bias* dan mematikan kelas netral.
* **Temuan 2: Jurang Out-of-Vocabulary (OOV) 70.5%:**
  * 70.49% kata di data uji sama sekali tidak ada di 300 data berlabel.
  * Korpus tanpa label menurunkan OOV drastis ke 18.15% (bukti empiris mutlak perlunya data tanpa label).
* **Karakteristik Teks:** Ulasan nyata berbahasa Indonesia majemuk (kuliner, transportasi, politik) dengan variasi slang dan singkatan tinggi.

### Slide 3: Metodologi & Arsitektur Semi-Supervised
* **Representasi Fitur:** Transductive Hybrid TF-IDF (Word (1,2) + Char (3,5)) untuk menangkap morfem slang dan frasa sentimen tanpa kebocoran label.
* **Base Learners:** Complement Naive Bayes (khusus data teks timpang) + Calibrated LinearSVC + Logistic Regression dalam skema *Soft Voting Ensemble*.
* **Inovasi Pencegahan Bias Konfirmasi:**
  * **Class-Balanced Adaptive Pseudo-Labeling (CB-PL):** Alokasi kuota berimbang per kelas (Top-$K$) agar pseudo-label tidak dimonopoli kelas positif.
  * **Sample Weighting ($w=0.5$):** Menjaga integritas data *ground truth* asli dari potensi derau (*noise*).
* **Validasi Ketat:** Stratified 5-Fold CV (Seed 42) yang steril dari kebocoran data.

### Slide 4: Hasil Komparasi, Diagnostik, & Keterbatasan
* **Tabel Komparasi Macro-F1 (CV):**
  * Supervised Baseline: Skor patokan dicatat fold-by-fold.
  * Semi-Supervised (CB-PL): Peningkatan Macro-F1 yang stabil lintas lipatan.
* **Diagnostik Pseudo-Label:**
  * Jumlah pseudo-label tersaring, proporsi distribusi antarkelas yang terkendali, dan rata-rata tingkat keyakinan $\ge 0.85$.
* **Keterbatasan & Analisis Derau:**
  * Ulasan bernada sarkasme atau ulasan politik ambigu tetap rentan salah prediksi jika keyakinan probabilitas terlalu dipaksakan tanpa kuota kelas.

### Slide 5: Rekomendasi Bisnis & Strategi Alokasi Anggaran (Active Learning)
* **Rekomendasi Implementasi:**
  * Gunakan pendekatan Semi-Supervised Hybrid sebagai solusi hemat biaya untuk menyaring opini konsumen secara otomatis.
* **Strategi Alokasi 300 Label Tambahan (Active Learning):**
  1. *Uncertainty/Margin Sampling:* Labeli sampel dengan entropi tertinggi (dekat batas keputusan batas netral/negatif).
  2. *Minority-Class Boosting:* Prioritaskan ulasan yang diduga kuat kelas Netral untuk menyeimbangkan distribusi.
  3. *Diversity Sampling:* Clustering embedding ulasan tanpa label untuk meliput topik-topik domain baru.

---

## 4. Skrip Pertahanan Tanya Jawab Juri (Durasi: 5 Menit)

Berikut adalah panduan jawaban resmi berbasis bukti untuk 3 pertanyaan dewan juri:

### Pertanyaan 1: "Mengapa Anda memakai metrik validasi itu?"
> **Jawaban:**  
> *"Kami memilih **Macro-F1** karena data berlabel memiliki ketimpangan kelas yang sangat ekstrem—58,3% positif berbanding hanya 10,3% netral. Jika kami menggunakan metrik Akurasi, model dapat 'curang' dengan hanya memprediksi positif dan tetap mendapatkan skor tinggi, namun gagal total pada ulasan netral. Macro-F1 memberikan bobot yang setara (33,3%) untuk masing-masing kelas tanpa memandang jumlah sampelnya. Untuk skema evaluasi, kami menerapkan **Stratified 5-Fold Cross-Validation** dengan split terkunci identik agar setiap perbandingan antara baseline dan semi-supervised bersifat adil (apples-to-apples) dan terbebas dari kebocoran data (data leakage)."*

### Pertanyaan 2: "Apa risiko metode semi-supervised yang Anda pilih dan bagaimana Anda memeriksanya?"
> **Jawaban:**  
> *"Risiko terbesar adalah **Confirmation Bias** dan **Semantic Drift**. Pada data yang imbalanced, model cenderung sangat percaya diri pada kelas mayoritas (positif) dan memasukkan pseudo-label positif dalam jumlah berlebih, yang kemudian memperparah ketimpangan dan menghancurkan performa kelas netral.  
> Kami memeriksa dan memitigasinya dengan tiga cara:  
> 1. Menerapkan **Class-Balanced Top-K Pseudo-Labeling**, di mana kuota pseudo-label dibatasi dan diseimbangkan per kelas, bukan flat global threshold.  
> 2. Menerapkan **Sample Weighting** ($w=0.5$) agar pseudo-label tidak mendominasi 300 data latih asli.  
> 3. Melakukan **Audit Diagnostik Kuantitatif** pada distribusi pseudo-label dan memverifikasi kenaikan Macro-F1 pada set validasi steril sebelum menghasilkan submission final."*

### Pertanyaan 3: "Kalau ada anggaran 300 label lagi, bagaimana Anda memilih datanya?"
> **Jawaban:**  
> *"Kami tidak akan memilih data secara acak (*random sampling*), melainkan menggunakan pendekatan **Active Learning** terarah:  
> 1. **Uncertainty Sampling (Margin Sampling):** Memilih ulasan tanpa label yang probabilitas prediksinya paling ambigu (probabilitas mendekati margin keputusan 0.33 antar-kelas) karena sampel ini memberikan *information gain* tertinggi bagi model.  
> 2. **Minority-Class Boosting:** Secara terarah memprioritaskan ulasan yang memiliki probabilitas tertinggi pada kelas `neutral` guna mengatasi defisit data netral yang saat ini hanya 31 sampel.  
> 3. **Diversity Sampling:** Melakukan clustering (misal K-Means) pada ruang representasi TF-IDF data tanpa label, lalu mengambil sampel dari klaster-klaster yang representasinya paling jauh dari 300 data latih awal untuk memperkaya topik baru (out-of-distribution coverage)."*
