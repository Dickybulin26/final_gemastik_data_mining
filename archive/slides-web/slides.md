# 5 Slide Presentasi — Final GEMASTIK Penambangan Data (Durasi: 5 Menit)

> **Status:** ✅ **FINAL — seluruh placeholder sudah diisi angka nyata hasil eksperimen** (sumber SSOT: `notebook.ipynb` Bagian 5–11 dan `progress-tracker.md` Bagian 2–3; `submission.csv` lolos `check_submission.py` → `[OK] submission.csv valid (1246 baris)`). Tidak ada angka karangan.
> **Pembagian waktu:** Slide 1 (45 dtk) · Slide 2 (60 dtk) · Slide 3 (75 dtk) · Slide 4 (75 dtk) · Slide 5 (60 dtk).

---

## Slide 1 — Masalah Bisnis & Keselarasan Tema SDG

**Judul:** Inovasi Klasifikasi Sentimen Hemat Biaya Berbasis Semi-Supervised Learning

**Angka kunci (strip atas):** `300 label` · `10.633 unlabeled` · `1.246 test` · `Macro-F1`

- **Masalah bisnis:** Anggaran industri hanya cukup menganotasi **300 ulasan** dari **10.633 ulasan mentah** — 2,8% data sisanya buta label.
- **Pertanyaan inti:** Apakah data tanpa label adalah **aset bernilai (boon)** atau justru **racun (bane)** bagi model — noisy & *confirmation bias*?
- **Jawaban singkat kami:** terukur **bernilai** — Macro-F1 naik dari baseline **0,6566** ke **0,7573** (5/5 lipatan) hanya dengan **200 pseudo-label** pilihan.
- **Solusi kami:** *Class-Balanced Pseudo-Labeling* (CB-PL) + *sample weighting* lemah (w=0,1) — latih ratusan ribu fitur TF-IDF dalam hitungan detik, nol risiko OOM.
- **SDG 8 (Decent Work):** Efisiensi biaya anotasi & optimasi beban kerja tim label manusia — 10.633 ulasan diproses otomatis, anggaran fokus pada 300 label bernilai tinggi.
- **SDG 9 (Industry & Innovation):** Infrastruktur AI adaptif yang mengubah data mentah tak berlabel skala besar menjadi inteligensi bisnis.

---

## Slide 2 — Temuan Kritis Eksplorasi Data (EDA)

**Judul:** Dua Temuan yang Membentuk Seluruh Desain Solusi

**Angka kunci (strip atas):** `58,3% / 31,3% / 10,3%` · `OOV 70,49% → 18,15%` · `~33 kata/ulasan`

- **Temuan 1 — Ketimpangan kelas ekstrem:** positive 175 (58,3%) · negative 94 (31,3%) · **neutral hanya 31 (10,3%)** → per fold latih hanya ±140 / ±75 / **±25** sampel → model naif akan mematikan kelas netral.
- **Temuan 2 — Jurang OOV masif:** **70,49%** jenis kata data uji tidak pernah ada di 300 data latih; bergabung dengan korpus unlabeled, OOV anjlok ke **18,15%** (kosa kata 2.355 → 17.838 unik) — bukti empiris mutlak nilai data tanpa label.
- **Karakteristik teks:** rata-rata ~33 kata di *ketiga* dataset (latih/unlabeled/uji) → satu distribusi sumber yang sama; multi-domain (kuliner, transportasi, politik) penuh slang (`bgt`, `gak`, `aja`).
- **Implikasi desain:** dibutuhkan representasi tahan-OOV (char n-gram + kosakata transductive) + mekanisme anti-*confirmation bias* yang eksplisit — bukan threshold global tunggal.

---

## Slide 3 — Metodologi & Arsitektur Semi-Supervised

**Judul:** Tiga Pilar: Representasi Transductive · Guru Terkalibrasi · CB-PL Anti-Bias

**Angka kunci (strip atas):** `TF-IDF word(1,2) + char(3,5)` · `teacher & model: Calibrated LinearSVC` · `Stratified 5-Fold, seed 42`

- **Pilar 1 — Representasi transductive:** kosakata TF-IDF di-fit pada gabungan **300 + 10.633 = 10.933 teks** (word 1–2 + char 3–5, ±110 rb fitur vs ±9 rb pada fitur supervised-only); sah & tanpa kebocoran label karena hanya memakai distribusi *unsupervised*, dan validation fold tidak ikut di-fit.
- **Pilar 2 — Guru terkalibrasi, bukan ensemble lemah:** kandidat baseline (LR, Complement NB, Soft Voting) diuji jujur; untuk pseudo-label dipakai **Calibrated LinearSVC** (Platt scaling) sebagai *teacher* **sekaligus** model akhir — satu-satunya model dengan probabilitas benar-benar terkalibrasi, sehingga ambang keyakinan τ bermakna. *Soft voting* justru meredam probabilitas (maksimum ±0,70), sehingga threshold tinggi menyerap nol sampel — terbukti saat diuji.
- **Pilar 3 — CB-PL anti-bias:** kuota **Top-K per kelas** (bukan threshold global tunggal) + ambang keyakinan τ dari grid **{0,80; 0,85; 0,90} × K=100/kelas** → terbaik **τ=0,85**, agar pseudo-label tidak dimonopoli kelas positive.
- ***Sample weighting* lemah:** data ground truth bobot 1,0 vs pseudo-label bobot **0,1** — bobot besar merusak (w=0,8 → 0,6059); dengan w=0,1 hanya **200 pseudo-label (1,88% pool)** tersaring, suara 300 label asli tidak pernah tenggelam.
- **Aturan seleksi terdaftar (pre-registered):** semua konfigurasi dinilai pada lipatan identik; model final = **mean Macro-F1 CV tertinggi** — bukan pilihan pascafakta.
- **Validasi steril:** Stratified 5-Fold CV (seed 42), fold validasi **tidak pernah** terpapar pseudo-label maupun penyetelan threshold.

---

## Slide 4 — Hasil Komparasi, Diagnostik, & Keterbatasan

**Judul:** Baseline vs Semi-Supervised — Angka, Diagnostik, dan Kejujuran Ilmiah

**Angka kunci (strip atas):** `Baseline 0,6566` → `SSL 0,7573` · `Δ +0,1007` · `menang 5/5 fold, p=0,022`

**Tabel komparasi Macro-F1 (Stratified 5-Fold CV, seed 42, lipatan identik):**

| Pendekatan | Macro-F1 (mean ± std) | Δ vs pembanding | Catatan |
|---|---|---|---|
| Dummy (tebak mayoritas) | 0,2456 ± 0,0000 | — | bukti akurasi menipu |
| Supervised-only terbaik (EXP-01, Logistic Regression) | 0,6396 ± 0,0471 | — | hanya 300 label, ±9 rb fitur |
| Ensemble transductive (EXP-04b) | 0,6476 ± 0,0918 | +0,0290 | vs supervised-only 0,6186 — hanya ganti kosakata |
| **Baseline terkuat (EXP-04a): Calibrated LinearSVC transductive** | **0,6566 ± 0,0533** | +0,0287 | vs EXP-02 (0,6279); patokan pembanding final |
| Self-training NAIF (EXP-05, top-20% keyakinan global) | 0,6278 ± 0,0665 | −0,0188 | vs EXP-04b — **menurun**; confirmation bias terbukti |
| CB-PL: teacher ENS → model ENS (EXP-06, τ=0,6; K=100; w=0,1) | 0,7129 ± 0,0443 | +0,0653 | vs EXP-04b — kuota per kelas bekerja |
| CB-PL: teacher ENS → model SVC (EXP-07) | 0,7460 ± 0,0689 | +0,0894 | vs EXP-04a (5/5 fold, p=0,040) |
| **MODEL FINAL (EXP-08): teacher SVC → model SVC (τ=0,85; K=100/kelas; w=0,1; 1 iterasi)** | **0,7573 ± 0,0549** | **+0,1007** | **vs EXP-04a — menang 5/5 fold, p=0,022** |

- **Skor per fold:** EXP-04a [0,6371; 0,6865; 0,6117; 0,7456; 0,6021] vs EXP-08 [0,6973; 0,8483; 0,7718; 0,7660; 0,7031] — selisih positif di **kelima** lipatan.
- **Diagnostik pseudo-label (kualitas > kuantitas):**
  - *Self-training naif (EXP-05):* menyerap **±2.127 pseudo-label per fold (±20% pool)** dengan komposisi **99,0% positive / 1,0% negative / 0,0% neutral**, rata-rata keyakinan hanya **0,664** → Macro-F1 justru turun **0,6476 → 0,6278**. Confirmation bias bukan asumsi, tapi temuan empiris.
  - *CB-PL final (EXP-08):* hanya **200 pseudo-label (1,88% pool)** — komposisi **50,0% positive / 50,0% negative / 0,0% neutral**, rata-rata keyakinan **±0,85** → sedikit tapi bersih.
- **Ablasi (jujur):** seleksi global tanpa kuota kelas = **0,5970** (jatuh di bawah baseline); bobot besar **w=0,8 → 0,6059**; **2 iterasi → 0,7008** (tak lebih baik); CB-PL pada fitur supervised-only = **0,6343** (pseudo-label butuh ruang fitur transductive); grid τ maks **0,7573 tepat di τ=0,85**.
- **Inferensi akhir:** `submission.csv` **1.246 baris** lolos `[OK]`; distribusi prediksi **positive 750 (60,2%) / negative 350 (28,1%) / neutral 146 (11,7%)** — kelas netral tidak punah (data latih 10,3%).
- **Keterbatasan (jujur):** p=0,022 pada 5 lipatan tetap estimasi kasar; seleksi hyperparameter (τ, K, w, teacher) pada CV yang sama berpotensi *optimistic*; ulasan sarkastik & politik ambigu tetap rentan salah kelas; **pseudo-label `neutral` ≈ 0% di semua konfigurasi** → kelas netral belum terpecahkan, dialihkan ke Active Learning.

---

## Slide 5 — Rekomendasi Bisnis & Strategi Alokasi Anggaran (Active Learning)

**Judul:** Dari Eksperimen ke Anggaran: Berikut 300 Label Berikutnya Dipakai untuk Apa?

**Angka kunci (strip atas):** `300 label berikutnya` · `OOV 18,15% tersisa` · `SDG 8 & 9`

- **Rekomendasi implementasi:** gunakan pipeline **semi-supervised hybrid** (0,6566 → 0,7573) sebagai penyaring opini konsumen otomatis hemat biaya; label manusia dialihkan ke titik dengan nilai informasi tertinggi.
- **Strategi alokasi 300 label berikutnya (Active Learning):**
  1. ***Uncertainty / Margin Sampling*** — labeli ulasan dengan entropi tertinggi (dekat margin keputusan negatif vs netral).
  2. ***Minority-Class Boosting*** — prioritaskan ulasan berprobabilitas tinggi kelas `neutral` (saat ini hanya 31 sampel; pseudo-label ≈ 0%).
  3. ***Diversity Sampling*** — K-Means pada representasi TF-IDF data unlabeled untuk meliput topik/domain baru yang belum tercakup.
- **Mitigasi lanjutan:** *human-in-the-loop* untuk sampel ambig + pemantauan *drift* berkala pada pseudo-label produksi.
- **Penutup:** data tanpa label = leverage murah — terbukti dari OOV 70,49% → 18,15% dan Macro-F1 0,6566 → 0,7573 — asalkan dijaga oleh kuota kelas, bobot kecil (0,1), dan validasi yang steril. **Selaras SDG 8 & 9.**
