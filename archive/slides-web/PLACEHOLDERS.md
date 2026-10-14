# Daftar Placeholder Materi Presentasi — STATUS: SEMUA SUDAH TERISI ✅

> **Status akhir:** Seluruh placeholder pada [`slides.md`](slides.md) dan [`slides.html`](slides.html) **sudah dihapus dan diganti angka final** — tidak ada lagi token kurung kurawal tersisa di keduanya (verifikasi: 0 kecocokan). [`../defense_sheet.md`](../defense_sheet.md) juga sudah terisi angka final.
>
> **SSOT angka hasil:** `notebook.ipynb` (28 sel, eksekusi bersih — Bagian 5 baseline, Bagian 6 self-training naif, Bagian 7 CB-PL & ablasi, Bagian 8 diagnostik, Bagian 9 tabel komparasi, Bagian 10 inferensi, Bagian 11 kesimpulan) + `progress-tracker.md` (Bagian 2 log EXP-00–EXP-08, Bagian 2.1 ablasi, Bagian 3 diagnostik, Bagian 5 ringkasan juri).
> **Aturan kejujuran:** setiap nilai di bawah dikutip apa adanya dari sumber tersebut — tanpa perombakan, tanpa angka karangan.

## Tabel Nilai Final (semua placeholder sudah terisi)

| # | Nama placeholder (asli) | Makna | **Nilai final** | Sumber sel / tracker |
|---|---|---|---|---|
| 1 | `MACRO_F1_BEST_BASELINE` | Mean Macro-F1 5-Fold CV baseline **terbaik** | **0,6566** (EXP-04a, Calibrated LinearSVC transductive); baseline supervised-only terbaik = 0,6396 (EXP-01 LR) | notebook Bagian 5 (output tabel baseline) & Bagian 9; tracker Bagian 2 baris EXP-04a |
| 2 | `STD_BEST_BASELINE` | Simpangan fold pada baseline terbaik | **0,0533** (EXP-04a; simpangan EXP-01 = 0,0471) | notebook Bagian 5 & 9; tracker Bagian 2 |
| 3 | `BEST_BASELINE_NAME` | Nama model baseline terbaik | **EXP-04a — Calibrated LinearSVC (fitur transductive)**; kandidat terkuat lain: EXP-01 LR (supervised-only) & EXP-04b Soft Voting | notebook Bagian 5; tracker Bagian 2 |
| 4 | `MACRO_F1_BEST_SSL` | Mean Macro-F1 konfigurasi semi-supervised terbaik | **0,7573** (EXP-08 CB-PL) | notebook Bagian 9 (baris `MODEL FINAL TERPILIH`); tracker Bagian 2 baris EXP-08 |
| 5 | `STD_BEST_SSL` | Simpangan fold konfigurasi CB-PL terbaik | **0,0549** | notebook Bagian 9; tracker Bagian 2 |
| 6 | `DELTA_F1` | Selisih: SSL terbaik − baseline terbaik | **+0,1007** (0,7573 − 0,6566), menang **5/5 fold**, **p=0,022** | notebook Bagian 9 (output `Δ = +0.1007 ... p = 0.022`); tracker Bagian 2 |
| 7 | `FOLD_SCORES_BASELINE` | Skor per fold baseline terbaik (EXP-04a) | **[0,6371; 0,6865; 0,6117; 0,7456; 0,6021]** | notebook Bagian 9; tracker Bagian 2 |
| 8 | `FOLD_SCORES_SSL` | Skor per fold CB-PL terbaik (EXP-08) | **[0,6973; 0,8483; 0,7718; 0,7660; 0,7031]** | notebook Bagian 9; tracker Bagian 2 |
| 9 | `BEST_CONFIG` | Konfigurasi hiperparameter CB-PL final | **teacher = Calibrated LinearSVC → model = Calibrated LinearSVC, τ=0,85, K=100/kelas, w=0,1, iterasi=1 (EXP-08)** | notebook Bagian 9 (`FINAL_CFG`) & Bagian 10; tracker Bagian 2 & Fase 4 |
| 10 | `THRESHOLD` | Ambang keyakinan terpilih (grid 0,80 / 0,85 / 0,90) | **0,85** (tertinggi pada grid teacher SVC; 0,80 → 0,7383, 0,90 → 0,7432) | notebook Bagian 7.1 (output `Teacher terbaik`); tracker Bagian 2.1 |
| 11 | `PSEUDO_WEIGHT` | Bobot sampel pseudo-label (ground truth = 1,0) | **0,1** (w=0,15/0,2/0,3 → 0,7011; w=0,8 merusak → 0,6059) | notebook Bagian 7.1; tracker Bagian 2.1 |
| 12 | `N_PSEUDO` | Jumlah pseudo-label terserap saat pelatihan final | **200 sampel (1,88% dari pool 10.633)** | notebook Bagian 10 (output `pseudo-label terpakai`) & sel verifikasi Bagian 11; tracker Bagian 3 |
| 13 | `DISTRIBUSI_PSEUDO` | Distribusi kelas pseudo-label final vs GT | **positive 50,0% / negative 50,0% / neutral 0,0% (n=200)** vs GT 58,3 / 31,3 / 10,3 | notebook Bagian 10; tracker Bagian 3 baris EXP-08 |
| 14 | `AVG_CONFIDENCE` | Rata-rata probabilitas sampel pseudo yang lolos | **±0,85** (EXP-08); pembanding naif EXP-05 = **0,664** | notebook Bagian 10; tracker Bagian 3 |
| 15 | `KALIMAT_KESIMPULAN_AKHIR` | Kalimat penutup lisan pada `defense_sheet.md` (Kartu Cepat) | *"Data tanpa label terbukti membantu secara terukur — OOV 70,49%→18,15%, Macro-F1 0,6566→0,7573 (5/5 lipatan, p=0,022) — asalkan pseudo-label disaring ketat, diberi bobot kecil, dan diaudit; kelas netral justru belum terbantu, dan itu pekerjaan rumah Active Learning berikutnya."* | `defense_sheet.md` §Kartu Cepat (sudah terisi) + notebook Bagian 11 |

## Angka pendukung yang dipakai di slide (bukan placeholder, sudah final)

| Angka | Nilai | Sumber |
|---|---|---|
| Data latih / unlabeled / uji | 300 (175/94/31 = 58,3 / 31,3 / 10,3%) · 10.633 · 1.246 | AGENTS.md §2, notebook Bagian 1 |
| OOV type-level & kosa kata | 70,49% → 18,15%; 2.355 → 17.838 unik; ±33 kata/ulasan | notebook Bagian 3 (sel verifikasi Bagian 11) |
| Validasi | Stratified 5-Fold CV, seed 42, lipatan identik, validation fold steril | AGENTS.md §3, notebook Bagian 4 |
| Self-training naif (EXP-05) | 0,6278 ± 0,0665; ±2.127 pseudo/fold (±20% pool); 99,0% positive / 1,0% negative / 0,0% neutral; keyakinan 0,664 | tracker Bagian 2 & 3; notebook Bagian 6 & 8 |
| Ensemble transductive (EXP-04b) | 0,6476 ± 0,0918 (dari 0,6186 supervised-only, +0,0290) | notebook Bagian 5; tracker Bagian 2 |
| Ablasi kunci | global tanpa kuota 0,5970 · w=0,8 → 0,6059 · 2 iterasi 0,7008 · supervised-only 0,6343 · ENS→ENS 0,7129 · ENS→SVC 0,7460 | tracker Bagian 2.1; notebook Bagian 7.1–7.2 |
| Distribusi `submission.csv` final | positive 750 (60,2%) / negative 350 (28,1%) / neutral 146 (11,7%); 1.246 baris → `[OK]` | notebook Bagian 10–11; tracker Bagian 4 |

## Catatan kejujuran ilmiah

- Δ final **positif (+0,1007)**; temuan negatif tetap ditampilkan apa adanya: EXP-05 (self-training naif) **0,6278** lebih rendah dari baseline, seleksi global tanpa kuota **0,5970**, dan w=0,8 **0,6059** — bukti confirmation bias, bukan kegagalan metodologi (AGENTS.md §7).
- Kelas `neutral` nyaris absen dari pseudo-label (**0,0%** pada EXP-08; ≈0,4–0,5% pada CB-PL dengan τ lebih rendah) → rekomendasi dialihkan ke Active Learning (uncertainty, minority-boosting, diversity K-Means).
