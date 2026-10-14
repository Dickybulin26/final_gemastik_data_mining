# Hasil Eksperimen Modeling (Jalur A - Fase 3 s.d. 6)

> **⚠️ CATATAN REKONSILIASI (agent integrator):** Dokumen ini adalah **rangkaian eksperimen awal** (jalur paralel A). Setelah itu, rangkaian eksperimen yang **lebih lengkap & final** dijalankan di `notebook.ipynb` (baseline EXP-04a = 0,6566; model final **EXP-08 CB-PL = 0,7573 ± 0,0549**, τ=0,85, K=100/kelas, w=0,1; lihat `progress-tracker.md` §2). **Rujukan angka resmi untuk seluruh deliverable = `notebook.ipynb` + `progress-tracker.md`.** Bagian `submission` di bawah (distribusi 778/340/128) sudah **usang** — `submission.csv` final kini berisi 750/350/146 (hasil EXP-08).

Semua angka dihasilkan oleh `.venv/bin/python src/modeling/run_experiments.py` (eksekusi nyata, tanpa fabricasi). Protokol: **Stratified 5-Fold CV (shuffle, random_state=42)**, metrik **Macro-F1**, validation fold steril dari fitting vectorizer & pseudo-labeling.

## 1. Tabel Komparasi Performa

| ID | Eksperimen | Representasi | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean | Std | Runtime (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| EXP-00 | Dummy Majority Baseline | none | 0.2456 | 0.2456 | 0.2456 | 0.2456 | 0.2456 | **0.2456** | 0.0000 | 0.0 |
| EXP-01 | Logistic Regression (balanced, C=1) | inductive | 0.5704 | 0.6891 | 0.6228 | 0.6960 | 0.6199 | **0.6396** | 0.0471 | 4.1 |
| EXP-02 | Calibrated LinearSVC (balanced, C=0.5) | inductive | 0.5704 | 0.5586 | 0.6342 | 0.7436 | 0.6325 | **0.6279** | 0.0656 | 0.2 |
| EXP-03 | Complement Naive Bayes (norm=True) | inductive | 0.5669 | 0.5799 | 0.4657 | 0.5598 | 0.4888 | **0.5322** | 0.0459 | 0.0 |
| EXP-04 | Soft Voting Ensemble (LR+SVM+CNB) | transductive | 0.6503 | 0.6865 | 0.5107 | 0.7882 | 0.6021 | **0.6476** | 0.0918 | 8.4 |
| EXP-05 | Self-training naive (thr>=0.90 global) | transductive | 0.7120 | 0.7458 | 0.7593 | 0.7422 | 0.6577 | **0.7234** | 0.0363 | 2.4 |
| EXP-06A | CB-PL (thr>=0.85, top-30/kelas, w=0.5) | transductive | 0.6371 | 0.7394 | 0.6342 | 0.7289 | 0.6466 | **0.6772** | 0.0467 | 0.8 |
| EXP-06B | CB-PL (thr>=0.85, top-60/kelas, w=0.5) | transductive | 0.6822 | 0.8204 | 0.6942 | 0.7582 | 0.6484 | **0.7207** | 0.0612 | 0.8 |
| EXP-06C | CB-PL (thr>=0.90, top-30/kelas, w=0.5) | transductive | 0.6371 | 0.7394 | 0.6342 | 0.7289 | 0.6466 | **0.6772** | 0.0467 | 0.8 |
| EXP-06D | CB-PL (thr>=0.90, top-60/kelas, w=0.5) | transductive | 0.6822 | 0.7928 | 0.6942 | 0.7955 | 0.6466 | **0.7223** | 0.0607 | 0.8 |

* Baseline supervised terbaik: **EXP-01 - Logistic Regression (balanced, C=1)** (Macro-F1 0.6396).
* Metode semi-supervised terbaik: **EXP-05 - Self-training naive (thr>=0.90 global)** (Macro-F1 0.7234).
* **Data tanpa label MENAIKKAN performa**: metode semi-supervised terbaik (EXP-05) mengungguli baseline supervised terbaik (EXP-01) sebesar **+0.0838 Macro-F1** (0.7234 vs 0.6396).

## 2. Diagnostik Pemilihan Teacher Pseudo-Label (fold 0, 10.633 teks pool)

Temuan kritis yang mempengaruhi desain eksperimen: **soft-voting ensemble tidak pernah mencapai confidence 0.85 pada pool** (max-prob p99 = 0.700), sehingga self-training berbasis ensemble menyerap **NOL** pseudo-label. Penyebab: ComplementNB(norm=True) pada TF-IDF yang di-L1-normalisasi menghasilkan probabilitas ~uniform (max ~0.333), dan Logistic Regression class_weight=balanced meratakan probabilitas; rata-rata voting menekan keyakinan gabungan. Karena itu teacher pseudo-label EXP-05/06 adalah **Calibrated LinearSVC** (satu-satunya anggota dengan probabilitas terkalibrasi sungguhan).

| Kandidat teacher | p50 | p90 | p95 | p99 | Kandidat >= 0.85 | Kandidat >= 0.90 |
|---|---|---|---|---|---|---|
| logistic_regression_balanced | 0.538 | 0.693 | 0.727 | 0.779 | 2 | 0 |
| calibrated_linearsvc | 0.817 | 0.967 | 0.978 | 0.988 | 4745 | 3696 |
| complement_nb_norm | 0.333 | 0.333 | 0.333 | 0.333 | 0 | 0 |
| soft_voting_ensemble | 0.563 | 0.664 | 0.679 | 0.701 | 0 | 0 |

## 3. Diagnostik Pseudo-Label (Fase 5)

| Konfigurasi | Threshold | Top-K/kelas | Bobot | Pseudo tersaring (rata-rata fold) | % dari pool 10.633 | Distribusi kelas (%) pos/neg/neu | Rata-rata confidence | Confirmation bias? |
|---|---|---|---|---|---|---|---|---|
| EXP-05 | 0.9 | None | 1.0 | 3286 | 30.91% | 98.1 / 1.9 / 0.0 | 0.946 | YA (positif mendominasi) |
| EXP-06A | 0.85 | 30 | 0.5 | 60 | 0.57% | 49.7 / 49.7 / 0.7 | 0.958 | netral tertindas |
| EXP-06B | 0.85 | 60 | 0.5 | 120 | 1.13% | 49.8 / 49.8 / 0.3 | 0.953 | netral tertindas |
| EXP-06C | 0.9 | 30 | 0.5 | 60 | 0.56% | 50.0 / 50.0 / 0.0 | 0.959 | netral tertindas |
| EXP-06D | 0.9 | 60 | 0.5 | 112 | 1.06% | 54.0 / 46.0 / 0.0 | 0.958 | netral tertindas |

* **EXP-05 (self-training naive, threshold global >= 0.90 tanpa kuota):** Konfirmasi bias: proporsi pseudo 'positive' 98.1% jauh di atas ground truth 58.33%. Penindasan kelas netral: pseudo 'neutral' hanya 0.0% (ground truth 10.33%).
* **EXP-06D (Class-Balanced Pseudo-Labeling):** Penindasan kelas netral: pseudo 'neutral' hanya 0.0% (ground truth 10.33%). Namun, kuota top-K hanya menyeimbangkan pseudo positive/negative (masing-masing ~50%); pseudo `neutral` tetap nyaris nol (0.0%) karena guru model hampir tidak pernah memberi keyakinan >= 0.85 pada kelas netral.
* Ground truth (300 label): positive 58.33%, negative 31.33%, neutral 10.33%.

## 4. Analisis Jujur

1. **Data tanpa label MENAIKKAN performa**: metode semi-supervised terbaik (EXP-05) mengungguli baseline supervised terbaik (EXP-01) sebesar **+0.0838 Macro-F1** (0.7234 vs 0.6396).
2. Efek representasi transductive (adaptasi kosakata terhadap 10.633 ulasan tanpa label) terukur pada EXP-04 dengan Macro-F1 0.6476; ini menjawab temuan EDA bahwa OOV data uji anjlok dari 70,49% ke 18,15% ketika corpus tanpa label dimanfaatkan.
3. EXP-05 (naive) dan EXP-06D (CB-PL) praktis setara: selisih 0.0011 jauh di bawah simpangan fold (+/-0.061). EXP-05 menunjukkan confirmation bias pada KOMPOSISI pseudo-label (98.1% positif, 0.0% netral) namun - secara jujur - belum merugikan Macro-F1 pada 5 fold ini; CB-PL mencapai skor setara dengan komposisi pseudo yang jauh lebih sehat (positif/negatif ~50/50), yaitu perlindungan eksplisit bagi kelas netral tanpa biaya performa.
4. Kelas `neutral` (hanya 10,3% dari 300 label) tetap titik rapuh struktural: pada threshold >= 0.85 sekalipun, pseudo-label netral nyaris nol di semua konfigurasi CB-PL (0.0%) karena teacher hampir tidak pernah memberi keyakinan tinggi pada netral. Kuota top-K menyeimbangkan positif/negatif, bukan menambah netral - pendanaan label aktif berikutnya harus menyasar kelas netral.
5. EXP-06A dan EXP-06C identik fold-per-fold: pada top-30, 30 sampel paling yakin per kelas semuanya lolos threshold 0.90, sehingga ambang 0.85 vs 0.90 memilih sampel yang sama persis; perbedaan ambang baru berpengaruh pada top-60 (0.7207 vs 0.7223).
6. Keterbatasan: (a) keyakinan model pada ulasan sarkastik/ambigu tetap belum terkalibrasi sempurna; (b) noise pseudo-label pada domain campuran (kuliner, transportasi, politik) belum dipisah per topik; (c) angka CV dengan hanya 300 label memiliki simpangan fold yang wajar - dibaca sebagai estimasi, bukan kepastian.

## 5. Submission Akhir

* Konfigurasi final: **EXP-05 - Self-training naive (thr>=0.90 global)** (mean Macro-F1 CV 0.7234)
* Vectorizer final: transductive TF-IDF (110007 fitur); pseudo-labeling final: {'threshold': 0.9, 'top_k_per_class': None, 'pseudo_weight': 1.0, 'n_selected': 3597, 'pct_of_pool': 33.83, 'distribution': {'negative': 102, 'neutral': 0, 'positive': 3495}, 'distribution_pct': {'negative': 2.84, 'neutral': 0.0, 'positive': 97.16}, 'avg_confidence': 0.9488}
* Distribusi prediksi `submission.csv` (1.246 baris): {'positive': 778, 'negative': 340, 'neutral': 128} -> {'positive': 62.44, 'negative': 27.29, 'neutral': 10.27}
* Verifikasi `check_submission.py`: **[OK] /home/dicky/Downloads/G3_Semi_supervised/submission.csv valid (1246 baris)**

## 6. Runtime per Tahap (detik)

| Tahap | Runtime (s) |
|---|---|
| load_and_clean | 0.2 |
| vectorizer_cache | 21.7 |
| teacher_profile | 2.9 |
| experiments | 18.3 |
| final_and_submission | 4.8 |
| total | 48.1 |

Total runtime pipeline: **48.1 detik (0.8 menit)**.
