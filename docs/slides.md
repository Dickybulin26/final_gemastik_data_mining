# Slide Presentasi — GEMASTIK Penambangan Data (5 Slide, 5 Menit)

> Berkas pendamping `slides.pptx`. Setiap slide disertai *speaker notes* (catatan pidato) untuk simulasi tanya jawab.

---

## Slide 1 — Masalah Bisnis & Keselarasan Tema SDG

**Inovasi Klasifikasi Sentimen Hemat Biaya berbasis Semi-Supervised Learning**

Platform ulasan hanya mampu membiayai anotasi manual **300 ulasan**, padahal tersedia **10.633 ulasan mentah tanpa label** dan **1.246 ulasan uji**.

Pertanyaan inti: *Apakah data tanpa label merupakan aset bernilai (boon) atau justru racun (bane) bagi model?*

* **SDG 8 — Decent Work and Economic Growth:** anotasi manual mahal; semi-supervised mengubah 10.633 ulasan mentah menjadi nilai tanpa menambah biaya anotasi; tim anotasi difokuskan pada sampel paling informatif (*Active Learning*).
* **SDG 9 — Industry, Innovation and Infrastructure:** pipeline AI ringan (TF-IDF + model linier, tanpa GPU, inference < 1 detik) yang layak dipasang di infrastruktur produksi, dan sepenuhnya *reproducible* dalam satu notebook.

*Speaker notes:* Sebutkan angka 300 vs 10.633 sejak awal — itu hook bisnisnya. Tekankan bahwa pertanyaannya bukan "model apa yang paling canggih" tapi "bagaimana nilai bisnis dioptimalkan dengan anggaran tetap".

---

## Slide 2 — Temuan Kritis Eksplorasi Data

**Temuan 1: Ketimpangan Kelas Ekstrem**

| Kelas | Jumlah | Proporsi | Per fold latih (5-Fold CV) |
|---|---:|---:|---:|
| positive | 175 | 58,33% | ±140 |
| negative | 94 | 31,33% | ±75 |
| neutral | 31 | 10,33% | ±25 |

* Akurasi adalah metrik menyesatkan: model yang selalu menebak `positive` tetap mendapat akurasi 58%.
* Maka dipilih **Macro-F1** (bobot setara 33,3% per kelas).
* Risiko semi-supervised: **confirmation bias** mematikan kelas `neutral`.

**Temuan 2: Jurang Out-of-Vocabulary (OOV) 70,49%**

* 70,49% jenis kata pada data uji tidak pernah ada di 300 data berlabel.
* Dengan korpus 10.633 data tanpa label, OOV anjlok menjadi **18,15%**.
* Model supervised murni "buta leksikal" pada 7 dari 10 kata di data uji.

**Karakteristik teks:** multi-domain (kuliner, transportasi, politik/pilpres), slang & singkatan tinggi (`bgt`, `ga`, `gak`, `sy`, `yg`), panjang rata-rata ±29 kata seragam di ketiga set.

*Speaker notes:* Angka OOV 70,49% → 18,15% adalah bukti empiris terkuat; tanyakan ke juri "bagaimana model bisa benar pada kata yang tidak pernah ia lihat?".

---

## Slide 3 — Metodologi & Arsitektur Semi-Supervised

**Pilar 1 — Representasi Fitur Transductive:** TF-IDF gabungan Word n-gram (1,2) untuk frasa sentimen majemuk ("tidak enak", "sangat bagus") + Character n-gram (3,5) untuk slang/typo (`bgt`, `banget`, `gak`). Kosakata di-*fit* pada gabungan data latih + tanpa label (±110 ribu fitur) **tanpa menyentuh label** dan **tanpa validation fold** (bebas *data leakage*); data uji tidak dipakai dalam CV.

**Pilar 2 — Dua Tahap: Teacher & Classifier** untuk pseudo-label:

* **Teacher = Calibrated LinearSVC** (`class_weight='balanced'`) — satu-satunya model dengan probabilitas benar-benar terkalibrasi (Platt scaling), jadi ambang keyakinan τ bermakna.
* Soft voting ensemble justru **meredam** probabilitas (maksimum ±0,70), sehingga threshold tinggi menyerap nol sampel — terbukti saat diuji.
* **Classifier akhir = Calibrated LinearSVC** (margin maksimal, unggul pada fitur *sparse* berdimensi tinggi).
* Cepat & ringan: <30 detik, tanpa GPU, risiko OOM nol pada kompetisi.

**Pilar 3 — Class-Balanced Adaptive Pseudo-Labeling (CB-PL):**

* Kuota **Top-K per kelas**, bukan threshold global (anti-monopoli kelas mayoritas).
* Ambang keyakinan minimum τ ditetapkan dari **sebaran empiris** probabilitas pada pool tanpa label.
* **Sample weight pseudo = 0,1** — saran pseudo-label sengaja dibuat lemah agar data *ground truth* tetap dominan.
* Hasil akhir: hanya **200 pseudo-label (1,9%)** tersaring — disaring ketat, bukan membanjiri data asli.

**Validasi ketat:** Stratified 5-Fold CV (seed 42), lipatan identik untuk semua eksperimen, validation fold steril dari pseudo-labeling maupun tuning.

---

## Slide 4 — Hasil Komparasi, Diagnostik, & Keterbatasan

| Pendekatan | Macro-F1 (mean ± std) | Δ vs baseline | Catatan |
|---|---|---|---|
| Dummy (tebak mayoritas) | 0,2456 ± 0,0000 | — | bukti akurasi menipu |
| Supervised-only (±9 rb fitur) | 0,6186 ± 0,0569 | — | hanya 300 label |
| Transductive TF-IDF (±110 rb fitur) | 0,6476 ± 0,0918 | +0,0290 | hanya ganti kosakata |
| Self-training NAIF (top-20% global) | 0,6278 ± 0,0665 | −0,0188 | confirmation bias terbukti |
| CB-PL: teacher=ENS → model ENS (τ=0,6; K=100; w=0,1) | 0,7129 ± 0,0443 | +0,0653 | kuota per kelas bekerja |
| **CB-PL final: teacher=SVC → model SVC (τ=0,85; K=100; w=0,1)** | **0,7573 ± 0,0549** | **+0,1007** | **menang 5/5 fold, p=0,022** |

**Diagnostik pseudo-label:**

* **Self-training naif:** >98% pseudo-label berlabel `positive`, `neutral` 0% → confirmation bias terbukti empiris, dan skor justru turun (0,6476 → 0,6278).
* **CB-PL final:** hanya **200 pseudo-label (1,9% dari pool)** yang tersaring, terbagi 50% `positive` / 50% `negative`, rata-rata keyakinan ≈0,85. Sedikit tapi bersih.
* **Kelas `neutral` nyaris absen** dari pseudo-label (0%) — bukan karena dimonopoli mayoritas, melainkan karena keyakinan model pada kelas minoritas memang rendah. Justru inilah bukti jujur bahwa pseudo-labeling bukan obat untuk kelas netral.

**Keterbatasan (jujur):**

* 300 label membuat selisih CV sulit signifikan; p=0,022 pada 5 lipatan tetaplah estimasi kasar.
* Seleksi hyperparameter (τ, K, w, pilihan teacher) pada CV yang sama berpotensi *optimistic*.
* Ulasan sarkastik dan politik ambigu tetap rentan salah kelas.
* Kelas `neutral` (10,3%) belum terpecahkan oleh pseudo-labeling — butuh Active Learning.

---

## Slide 5 — Rekomendasi Bisnis & Strategi Alokasi Anggaran

**Rekomendasi implementasi:** adopsi pipeline semi-supervised hybrid sebagai penyaring opini otomatis skala 10.000+ ulasan. Biaya per ulasan teranalisis turun drastis tanpa menambah tim anotasi (SDG 8), sekaligus memperluas cakupan leksikal industri (SDG 9).

**Strategi 300 label tambahan (Active Learning), bukan random sampling:**

1. **Uncertainty / Margin Sampling** — labeli ulasan dengan probabilitas paling ambigu antar kelas (informasi gain tertinggi).
2. **Minority-Class Boosting** — prioritaskan ulasan yang diduga kuat `neutral` (kini hanya 31 sampel).
3. **Diversity Sampling** — clustering K-Means pada ruang TF-IDF untuk menjangkau topik baru yang belum terwakili.

**Kunci jawaban juri:**

* *Mengapa Macro-F1?* Karena akurasi menipu pada data 58% positif; Macro-F1 memperlakukan ketiga kelas secara setara.
* *Risiko semi-supervised?* Confirmation bias — dibuktikan lewat eksperimen naif di Slide 4 (99% pseudo `positive`, skor turun), lalu dimitigasi kuota per kelas + bobot 0,1 + teacher terkalibrasi + audit CV fold-by-fold.
* *300 label lagi?* Active Learning terarah, bukan pengambilan acak.
