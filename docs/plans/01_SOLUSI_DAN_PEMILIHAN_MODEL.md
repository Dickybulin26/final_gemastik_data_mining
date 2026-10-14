# 01. Solusi Teknis & Pemilihan Model yang Tepat

> **Tujuan Dokumen:** Menjelaskan secara komprehensif landasan teoritis, perbandingan kandidat arsitektur, pemilihan model terbaik, serta strategi penanganan risiko *confirmation bias* pada klasifikasi sentimen semi-supervised.

---

## 1. Analisis Anatomi Masalah & Tantangan Unik

Proyek ini memiliki kombinasi kendala ekstrem yang sering menggagalkan pendekatan machine learning konvensional:

1. **Ukuran Sampel Super Kecil (*Extreme Low-Resource Regime*):**
   * Hanya ada **300 data latih berlabel**.
   * Membagi 300 data ke dalam 5-Fold CV berarti setiap *train fold* hanya memiliki 240 sampel, dan *validation fold* hanya 60 sampel.
2. **Ketimpangan Kelas Ekstrem (*Severe Imbalance*):**
   * Positif: 175 (58.3%)
   * Negatif: 94 (31.3%)
   * Netral: 31 (10.3%)  $\rightarrow$ Hanya ada ~25 data netral per train-fold!
3. **Jurang Kosakata Masif (*Out-of-Vocabulary / OOV Gap*):**
   * 70.49% kata dalam data uji (`test.csv`) tidak ada di data latih berlabel.
   * Model supervised murni akan menjadi "buta leksikal" terhadap 7 dari 10 kata baru di data uji.
4. **Karakteristik Teks Indonesia Multi-Domain:**
   * Campuran topik kuliner, transportasi, fasilitas publik, dan perdebatan politik pilpres.
   * Tingkat variasi *slang*, kata singkatan (`bgt`, `ga`, `gak`, `sy`, `yg`), partikel lisan, dan imbuhan tidak baku sangat tinggi.

---

## 2. Mengapa Solusi Populer Tertentu GAGAL atau Berisiko Tinggi?

### A. Mengapa BUKAN Fine-Tuning Deep Learning (IndoBERT / LLM)?
Meskipun IndoBERT sering menjadi pilihan utama untuk NLP Indonesia, dalam konteks kompetisi ini model tersebut memiliki risiko kegagalan teknis fatal:
1. **Catastrophic Overfitting pada 300 Sampel:** Model dengan 110+ juta parameter yang di-fine-tune pada 240 sampel data latih sangat rentan mengalami memorisasi, instabilitas gradien, dan sensitivitas tinggi terhadap *random seed*.
2. **Batasan Komputasi & Waktu (60 Menit):** Membutuhkan download bobot model (~500MB - 1GB), instalasi PyTorch/Transformers (~2GB), dan pelatihan epoch lambat pada laptop GPU VRAM 4GB (RTX 3050).
3. **Kompleksitas & Risiko Crash:** Out-Of-Memory (OOM) runtime dapat menghabiskan kuota waktu simulasi 60 menit tanpa hasil yang dapat diserahkan.

### B. Mengapa BUKAN Naive Self-Training (Scikit-Learn Standar)?
`SelfTrainingClassifier` standar di Scikit-learn menggunakan *fixed threshold* tunggal (misal `threshold=0.85`).
* **Masalah Fatal (Confirmation Bias):** Model yang di-fit pada data imbalanced akan memiliki keyakinan probabilitas (*softmax confidence*) yang jauh lebih tinggi untuk kelas mayoritas (`positive`) dibandingkan kelas minoritas (`neutral`).
* Akibatnya: Ribuan ulasan tanpa label akan diprediksi sebagai `positive`, sementara kelas `neutral` hampir tidak pernah terpilih.
* Data latih yang terkontaminasi pseudo-label positif akan semakin menekan prediksi kelas netral pada evaluasi. F1-score untuk kelas netral akan anjlok mendekati 0, dan **Macro-F1 secara keseluruhan akan jatuh bebas**.

---

## 3. Solusi Tepat: *Tri-Strategy Hybrid Semi-Supervised Architecture*

Untuk memaksimalkan Macro-F1 secara stabil, cepat, dan terbukti ilmiah, kita merancang arsitektur 3 pilar:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   Pilar 1: Feature Representation                      │
│     Transductive Hybrid TF-IDF (Word n-gram 1-2 + Char n-gram 3-5)     │
│       - Kosakata di-fit pada (Labeled + Unlabeled Corpus: 10.933 teks) │
│       - Memotong OOV Rate dari 70.49% ke 18.15% seketika!              │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│                       Pilar 2: Model Ensemble                          │
│   Blended Probability Classifier dengan Penanganan Ketimpangan Kelas:  │
│   1. Complement Naive Bayes (CNB) -> Dirancang matematis untuk imbal.  │
│   2. Calibrated LinearSVC (class_weight='balanced') -> Margin optimal  │
│   3. Logistic Regression (class_weight='balanced', C=1.0)              │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│            Pilar 3: Class-Balanced Adaptive Pseudo-Labeling            │
│   - Kuota Terpisah per Kelas (Top-K per class) atau Adaptive Threshold │
│   - Mengamankan kuota pseudo-label kelas Netral (Anti-Confirmation)    │
│   - Sample Weighting: w_pseudo = 0.5 vs w_ground_truth = 1.0           │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Penjelasan Rinci Komponen Arsitektur

### 1. Pilar Fitur: Transductive Hybrid TF-IDF
* **Word n-gram (1, 2):** Menangkap unigram kata dan frasa sentimen majemuk (contoh: *"sangat puas"*, *"tidak enak"*, *"kurang ramah"*).
* **Character n-gram (3, 5):** Bekerja pada level subkata. Sangat ampuh mengatasi *slang*, singkatan, dan tipo Indonesia (contoh: *"bangeeet"*, *"bgt"*, *"gaaak"*, *"gak"* memiliki sub-irisan karakter identik).
* **Transductive Vocabulary Fitting:**
  * Kamus TF-IDF dibentuk dari gabungan teks `labeled.csv` + `unlabeled.csv`.
  * **Mengapa ini sah dan tidak bocor (*no leakage*)?** Dalam paradigma Semi-Supervised/Transductive Learning, teks tanpa label memang diasumsikan tersedia untuk mempelajari manifold ruang leksikal (*unsupervised distribution*). Fitur ini murni mengekstrak frekuensi kata tanpa pernah menyentuh atau mengetahui label target.

### 2. Pilar Model: Linear & Probabilistic Learners
Model-model ini dipilih karena:
1. **Kecepatan Komputasi Kilat (< 5 detik per pelatihan):** Sangat ideal untuk simulasi kompetisi 60 menit dan iterasi cepat.
2. **Kinerja Superior pada High-Dimensional Sparse Text:** Karakteristik TF-IDF dengan puluhan ribu fitur linier sangat cocok dipisahkan oleh *convex linear classifiers*.
3. **Pilihan Model Spesifik:**
   * **Complement Naive Bayes (CNB):** Dikembangkan khusus oleh Rennie dkk. (2003) untuk mengatasi kelemahan Multinomial Naive Bayes pada korpus teks yang timpang (*skewed class distribution*).
   * **Logistic Regression (`class_weight='balanced'`):** Menyesuaikan bobot loss secara terbalik dengan frekuensi kelas: $w_c = \frac{N}{|C| \cdot N_c}$. Menghasilkan estimasi probabilitas posterior $P(y|x)$ yang halus.
   * **CalibratedClassifierCV(LinearSVC):** SVM mencari hyperplane pemisah margin maksimal. Dikalibrasi dengan sigmoid (Platt Scaling) untuk menghasilkan probabilitas output yang akurat.
   * **Soft Voting Blending:** Merata-ratakan probabilitas dari ketiga model:
     $$P_{ensemble}(y=c|x) = \frac{1}{3} \left[ P_{CNB}(c|x) + P_{LR}(c|x) + P_{SVC}(c|x) \right]$$
     Ensembling secara drastis mengurangi *variance* dan mencegah model terjebak pada bias model tunggal.

### 3. Pilar Semi-Supervised: Class-Balanced Adaptive Pseudo-Labeling (CB-PL)
Untuk mencegah *Confirmation Bias*, algoritma pseudo-labeling dimodifikasi secara ketat:
* **Mekanisme Kuota / Top-K per Kelas:**
  * Misal kita ingin menambahkan $K = 150$ sampel per iterasi.
  * Alih-alih mengambil 150 sampel dengan probabilitas tertinggi secara global (yang hampir pasti 90% positif), kita menetapkan kuota:
    * Ambil Top-$K_{pos}$ ulasan untuk kelas `positive` (misal 50 sampel).
    * Ambil Top-$K_{neg}$ ulasan untuk kelas `negative` (misal 50 sampel).
    * Ambil Top-$K_{neu}$ ulasan untuk kelas `neutral` (misal 50 sampel).
  * Dengan cara ini, kelas `neutral` sengaja disuntik data baru berkeyakinan tertinggi, memperluas batas keputusan (*decision boundary*) kelas minoritas tanpa tertindas oleh kelas mayoritas.
* **Sample Weighting ($w$):**
  * Data asli (300 sampel) diberi bobot $w_{train} = 1.0$.
  * Data pseudo-label diberi bobot $w_{pseudo} = 0.5$ (atau proporsional dengan confidence score). Ini memastikan sampel pseudo-label tidak pernah mendikte atau merusak representasi data *ground truth*.

---

## 5. Ringkasan Matriks Perbandingan Model

| Kriteria Evaluasi | IndoBERT Fine-Tuning | Naive Self-Training (Standard) | **CB-PL + Transductive TF-IDF (Pilihan Kita)** |
|---|:---:|:---:|:---:|
| **Kesesuaian Data 300 Sampel** | Rendah (Overfitting) | Rendah (Bias Positif) | **Sangat Tinggi (Linear Regularized)** |
| **Ketahanan thd Class Imbalance** | Sedang | Sangat Buruk (Drift) | **Sangat Tinggi (Balanced Kuota & Weights)** |
| **Mitigasi OOV (70.5% $\rightarrow$ 18.2%)** | Tinggi | Rendah | **Sangat Tinggi (Corpus Vocabulary)** |
| **Kecepatan Training (60 Menit)** | Lambat (~15-30 menit) | Cepat (< 1 menit) | **Sangat Cepat (< 30 detik)** |
| **Risiko Runtime Crash / OOM** | Tinggi | Sangat Rendah | **Nol / Sangat Stabil** |
| **Kemudahan Interpretasi Juri** | Sulit (*Black Box*) | Rentan Dikritik | **Sangat Elegan & Ilmiah (Dapat Dibuktikan)** |

---

## 6. Kesimpulan Pemilihan Model
Pendekatan **Transductive Hybrid TF-IDF + Calibrated Balanced Ensemble + Class-Balanced Pseudo-Labeling (CB-PL)** adalah strategi paling rasional, tangguh (*robust*), dan ilmiah untuk memenangkan kompetisi ini dengan skor Macro-F1 tertinggi dan pertahanan argumen yang kokoh di hadapan dewan juri.
