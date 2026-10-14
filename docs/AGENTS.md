# AGENTS.md - Panduan Utama Proyek Semi-Supervised Learning (GEMASTIK Penambangan Data)

> **Dokumen ini adalah *Single Source of Truth (SSOT)* dan panduan operasional utama untuk agen AI dan pengembang dalam mengeksekusi proyek kompetisi Data Mining ini.**

---

## 1. Ringkasan Eksekutif & Konteks Kompetisi

* **Format Kompetisi:** Live Coding Machine Learning bergaya Final GEMASTIK Penambangan Data.
* **Batas Waktu Simulasi:** 60 menit pengerjaan teknis, presentasi 5 menit, tanya jawab (Q&A) 5 menit.
* **Tema Utama:** *Inovasi yang hemat biaya (Cost-effective Innovation)* berakar pada:
  * **SDG 8:** *Decent Work and Economic Growth* (Efisiensi biaya dan optimasi sumber daya manusia pada anotasi data ulasan konsumen).
  * **SDG 9:** *Industry, Innovation, and Infrastructure* (Infrastruktur kecerdasan buatan adaptif yang mampu mengekstraksi nilai bisnis dari data mentah tak berlabel skala besar).
* **Studi Kasus:** Sebuah platform ulasan hanya memiliki anggaran untuk menganotasi **300 ulasan** secara manual, sementara terdapat puluhan ribu ulasan mentah (**10.633 ulasan**) yang belum berlabel. Pertanyaan inti penguji: *Apakah pemanfaatan data tanpa label benar-benar membantu meningkatkan performa klasifikasi sentimen, atau justru merugikan (noisy/confirmation bias)?*

---

## 2. Struktur Data & Karakteristik Domain

Data berada di direktori `data/` dengan ringkasan sebagai berikut:

| Nama File | Jumlah Baris | Kolom | Keterangan & Karakteristik |
|---|---|---|---|
| `labeled.csv` | 300 | `id`, `text`, `label` | Data latih berlabel (*ground truth*). 3 kelas sentimen: `positive`, `negative`, `neutral`. |
| `unlabeled.csv` | 10.633 | `id`, `text` | Data tanpa label (*pool data*) untuk pembelajaran semi-supervised. |
| `test.csv` | 1.246 | `id`, `text` | Data evaluasi uji tersembunyi (label disimpan penguji). |
| `submission_template.csv` | 1.246 | `id`, `label` | Format baku keluaran prediksi untuk verifikasi. |

### Temuan Kritis Eksplorasi Data (EDA Findings):
1. **Ketimpangan Kelas Ekstrem (*Severe Class Imbalance*) pada 300 Data Latih:**
   * `positive`: 175 sampel (**58.33%**)
   * `negative`: 94 sampel (**31.33%**)
   * `neutral`: 31 sampel (**10.33%**)
   * *Implikasi:* Kelas mayoritas sangat mendominasi. Metode semi-supervised standar rentan mengalami **Confirmation Bias**, di mana model cenderung hanya mempercayai prediksi positif dan mengabaikan netral.
2. **Karakteristik Teks & Domain:**
   * Teks ulasan berbahasa Indonesia riil dari media sosial dan platform review.
   * Mengandung percampuran topik majemuk: ulasan kuliner/restoran, transportasi/fasilitas publik, serta opini sosial-politik (pilpres, tokoh, kebijakan).
   * Mengandung variasi kata informal, singkatan, slang, partikel bahasa lisan (`banget`, `bgt`, `ga`, `gak`, `aja`, `nya`), tanda baca berulang, dan teks sensitif.
   * Rata-rata panjang kata sangat konsisten di ketiga dataset: ~33 kata (Q1: 16-17 kata, Median: 28-29 kata, Q3: 47-48 kata, Maks: ~100 kata). Ini membuktikan data latih, tanpa label, dan uji berasal dari distribusi sumber yang sama.
3. **Out-of-Vocabulary (OOV) Rate yang Masif:**
   * Kosakata data latih (300 sampel): 2.355 kata unik.
   * Kosakata data uji (1.246 sampel): 5.516 kata unik.
   * **70.49% kata dalam data uji TIDAK PERNAH muncul di data latih berlabel!**
   * Ketika digabungkan dengan 10.633 data tanpa label (17.838 kata unik), OOV Rate langsung anjlok dari **70.49% menjadi hanya 18.15%**.
   * *Justifikasi Ilmiah:* Ini adalah bukti kuantitatif terkuat bahwa representasi berbasis data tanpa label sangat krusial untuk memperluas cakupan leksikal model.

---

## 3. Metrik Evaluasi & Protokol Validasi

### A. Metrik Utama: Macro-F1 Score
$$\text{Macro-F1} = \frac{1}{|C|} \sum_{c \in C} F1_c$$
* **Mengapa Macro-F1?**
  * Macro-F1 memberikan bobot yang setara (33.3%) untuk masing-masing kelas (`positive`, `negative`, `neutral`) tanpa memandang jumlah sampel.
  * Mencegah model "curang" dengan hanya memprediksi kelas mayoritas (`positive`). Model yang mengorbankan kelas minoritas (`neutral`) akan dihukum berat pada metrik ini.

### B. Protokol Validasi: Stratified 5-Fold Cross-Validation
* **Strategi:** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
* **Aturan Mutlak Pencegahan Data Leakage:**
  * Pembagian lipatan (*folds*) harus dipatok identik untuk semua eksperimen (baseline vs semi-supervised).
  * Pada setiap fold validasi, model semi-supervised HANYA boleh menggunakan data latih berlabel pada lipatan tersebut + data tanpa label. **Lipatan validasi (*validation fold*) tidak boleh terpapar pada proses pseudo-labeling atau penyesuaian threshold!**
  * Seluruh preprocessing dan representasi fitur (vektorisasi) harus di-fit hanya pada train set (+ pool unlabeled jika unsupervised/transductive), bukan pada set validasi.

---

## 4. Arsitektur Pemodelan & Peta Jalan Eksperimen

### Tahap 1: Preprocessing & Ekstraksi Fitur
* Pembersihan teks cerdas: normalisasi huruf kecil (*lowercasing*), penanganan tanda baca berlebih, standardisasi spasi, pelestarian emoticon/tanda sentimen penting.
* Ekstraksi Fitur:
  * **Word-level TF-IDF:** n-gram (1, 2) untuk menangkap frasa sentimen seperti "tidak enak", "sangat bagus".
  * **Character/Subword-level TF-IDF:** n-gram (3, 5) untuk menangkap variasi slang, tipo, dan morfem imbuhan Indonesia.
  * **Vocabulary Expansion:** Pemanfaatan corpus tanpa label untuk membentuk kamus TF-IDF lengkap (mengurangi OOV dari 70% ke 18%).
  * **Dense Dimensionality Reduction (Opsional):** TruncatedSVD (LSA) untuk menangkap fitur topik laten lintas domain.

### Tahap 2: Supervised Baseline (Hanya 300 Label)
* Algoritma pembanding:
  * Logistic Regression (dengan regularisasi $C$, class weight balanced).
  * Linear Support Vector Machine (LinearSVC / SGDClassifier).
  * Complement Naive Bayes (dirancang khusus untuk teks dengan ketimpangan kelas).
  * Ensembling / Soft Voting Classifier.
* Catat Macro-F1 rata-rata dan standar deviasi per fold sebagai batas patokan (*benchmark*).

### Tahap 3: Metode Semi-Supervised Learning
Eksplorasi dan bandingkan pendekatan berikut:
1. **Confidence-Thresholded Self-Training (Pseudo-Labeling):**
   * Latih model pada data berlabel.
   * Prediksi probabilitas pada data tanpa label.
   * Pilih sampel dengan keyakinan tinggi ($\max P(y|x) \ge \tau$, misal $\tau \in [0.85, 0.95]$).
   * **Pencegahan Confirmation Bias:** Gunakan **Class-Balanced Thresholding** atau kuota proporsional per kelas agar pseudo-label tidak dimonopoli oleh kelas `positive`.
   * Tambahkan pseudo-labeled samples ke set latih dan latih ulang model (*iterative self-training*).
2. **Label Spreading / Label Propagation (Graph-Based):**
   * Memanfaatkan manifold struktur data teks dengan k-NN graph similarity.
3. **Semi-Supervised Feature Representation (Unsupervised Pre-training / SVD):**
   * Menggunakan representasi ruang fitur yang di-fit secara transductive pada gabungan teks berlabel + tanpa label.

### Tahap 4: Diagnostik & Verifikasi Kualitas Pseudo-Label
Wajib melakukan audit mendalam terhadap pseudo-label:
* Berapa banyak data tanpa label yang tersaring ke dalam training?
* Bagaimana distribusi kelas pseudo-label dibanding distribusi ground truth?
* Analisis apakah terjadi *semantic drift* atau *error propagation* (konfirmasi bias).
* Bandingkan Macro-F1 secara jujur: Apakah penambahan pseudo-label menaikkan atau menurunkan skor validasi?

### Tahap 5: Inferensi Akhir & Verifikasi File Pengiriman
* Latih model terbaik pada seluruh data berlabel (+ data tanpa label terpilih).
* Hasilkan prediksi untuk `data/test.csv`.
* Simpan dalam format `submission.csv` (kolom `id` dan `label`).
* Jalankan verifikasi mutlak: `python check_submission.py submission.csv`. Pastikan status `[OK]`.

---

## 5. Standar Keluaran & Deliverables Proyek

Proyek ini harus menghasilkan luaran terstruktur berikut:

1. **`submission.csv`**:
   * File hasil prediksi akhir untuk 1.246 data uji.
   * Lolos 100% verifikasi `check_submission.py` (tanpa nilai null, id persis, label valid).
2. **`notebook.ipynb`**:
   * Satu file notebook mandiri (*single reproducible notebook*).
   * Struktur naratif terstruktur:
     1. Latar Belakang & Keselarasan SDG 8 & 9.
     2. Eksplorasi Data & Bukti OOV Rate.
     3. Strategi Validasi (Stratified 5-Fold).
     4. Model Baseline (Supervised).
     5. Implementasi Semi-Supervised Learning.
     6. Tabel Komparasi Performa (Macro-F1 baseline vs semi-supervised).
     7. Diagnostik Pseudo-Label (Kuantitas, Distribusi, Keyakinan).
     8. Kesimpulan Akhir (Maksimal 5 kalimat yang jujur dan ilmiah).
3. **5 Slide Presentasi (Format Ringkas & Berbobot):**
   * **Slide 1:** Judul, Masalah Bisnis, & Keselarasan Tema SDG (SDG 8 & 9).
   * **Slide 2:** Temuan Eksplorasi Data (Ketimpangan Kelas & Jurang OOV 70.5%).
   * **Slide 3:** Metodologi & Arsitektur Semi-Supervised (Pencegahan Bias Konfirmasi).
   * **Slide 4:** Hasil Komparasi & Keterbatasan (Macro-F1, Kualitas Pseudo-Label).
   * **Slide 5:** Rekomendasi Bisnis & Strategi Alokasi Anggaran (Active Learning).
4. **`progress-tracker.md`**:
   * Log pelacakan progres real-time, status task, dan hasil pencatatan eksperimen tiap fold.

---

## 6. Strategi Pertahanan Pertanyaan Juri (Juror Defense Sheet)

| Pertanyaan Juri | Jawaban Ilmiah & Strategis Berbasis Bukti |
|---|---|
| **1. Mengapa Anda memakai metrik validasi itu?** | "Kami memilih **Macro-F1** karena data berlabel mengalami ketimpangan kelas yang tajam (58.3% positif vs 10.3% netral). Akurasi standar akan membiaskan model untuk mengabaikan kelas netral demi skor tinggi semu. Macro-F1 memperlakukan ketiga kelas secara setara, memastikan keandalan model pada setiap kategori ulasan. Evaluasi dilakukan dengan **Stratified 5-Fold CV** dengan lipatan identik agar bebas dari kebocoran data (*data leakage*)." |
| **2. Apa risiko metode semi-supervised yang Anda pilih dan bagaimana Anda memeriksanya?** | "Risiko terbesar adalah **Confirmation Bias** dan **Semantic Drift** (perambatan kesalahan), di mana model overconfidence memberi pseudo-label salah pada ulasan ambigu sehingga merusak ruang fitur, khususnya memperparah ketimpangan kelas positif. Kami memitigasinya dengan: (1) Ambang batas keyakinan ketat ($\ge 0.90$), (2) *Class-balanced sampling* pada pseudo-label, dan (3) Audit diagnostik kuantitatif terhadap distribusi kelas pseudo-label serta komparasi ketat fold-by-fold CV sebelum memutuskan model final." |
| **3. Kalau ada anggaran 300 label lagi, bagaimana Anda memilih datanya?** | "Kami tidak akan mengambil data secara acak (*random sampling*), melainkan menerapkan **Active Learning**: (1) *Uncertainty Sampling* (memilih ulasan dengan entropi prediksi tertinggi / probabilitas dekat batas margin keputusan antarkelas), (2) *Minority-Boosting Sampling* (memprioritaskan ulasan yang diduga kuat kelas `neutral`), dan (3) *Diversity Sampling* (clustering k-means pada representasi embedding data tanpa label untuk meliput topik ulasan baru yang belum ada di 300 label awal)." |

---

## 7. Aturan Operasional & Lingkungan Teknis

* **Python Interpreter:** Wajib menggunakan virtual environment lokal proyek (`.venv/bin/python`).
* **Integritas Kode:** Semua sel notebook harus dapat dijalankan secara runtut dari atas ke bawah tanpa error (*clean execution*).
* **Kejujuran Ilmiah:** Jika data tanpa label ternyata tidak memberikan kenaikan signifikan atau mengalami degradasi akibat noise, nyatakan secara transparan disertai analisis penyebab ilmiahnya. GEMASTIK menghargai penalaran kritis dan temuan negatif yang valid secara metodologis.
