# EDAMEMO — Ringkasan EDA Jalur B (Fase 1: EDA & Bukti OOV)

> **Sumber angka:** `results/eda/eda_stats.json` (dihasilkan otomatis oleh `.venv/bin/python src/eda/run_eda.py`).
> Semua angka di bawah adalah **hasil hitungan langsung** pada data, bukan salinan teori.
> Tokenisasi acuan angka EDA: `str.lower().split()` — sama dengan metodologi temuan Bagian 2 `AGENTS.md`.
> Angka target AGENTS.md **terkonfirmasi 100% persis** (lihat §3).

---

## 1. Integritas Data (n=300 / 10.633 / 1.246)

* `labeled.csv`: **300 baris**, kolom `id, text, label` — **0 null** di semua kolom, **0 baris duplikat**.
* `unlabeled.csv`: **10.633 baris**, kolom `id, text` — **0 null**, **0 duplikat**.
* `test.csv`: **1.246 baris**, kolom `id, text` — **0 null**, **0 duplikat**.
* Total token: labeled **10.034**, unlabeled **351.417**, test **40.839** kata.
  → Rasio token unlabeled:labeled = **35:1** (ketersediaan data mentah sangat melimpah).

## 2. Ketimpangan Kelas pada 300 Data Latih (fig `01_class_distribution.png`)

| Kelas | Jumlah | Persen |
|---|---|---|
| positive | **175** | **58.33%** |
| negative | **94** | **31.33%** |
| neutral | **31** | **10.33%** |

* Rasio kelas mayoritas:minoritas = **175 : 31 ≈ 5.6 : 1**.
* *Implikasi narasi:* model yang menebak `positive` terus-menerus sudah mendapat akurasi 58.33% tanpa belajar apa pun — di sinilah **Macro-F1** wajib dipakai, dan pseudo-labeling tanpa pengendalian kelas akan memperparah dominasi `positive` (**confirmation bias**).

## 3. Profil Panjang Kata — Bukti Satu Distribusi Sumber (fig `02_word_length_hist.png`)

| Dataset | Mean | Q1 | Median | Q3 | Maks |
|---|---|---|---|---|---|
| labeled | **33.45** | 17 | 29 | 47 | 96 |
| unlabeled | **33.05** | 16 | 28 | 48 | 110 |
| test | **32.78** | 16 | 28 | 47 | 100 |

* Selisih mean antar dataset hanya **0.67 kata** (±1 std ≈ 21 kata di ketiga dataset).
* *Implikasi:* ketiga dataset berasal dari **domain & sumber yang sama** — aman menggunakan data tanpa label sebagai representasi fitur transductif untuk data uji.

## 4. Jurang OOV: Bukti Kuantitatif Utama Nilai Data Tanpa Label (fig `03_oov_comparison.png`)

Kosakata unik:
* labeled = **2.355** | unlabeled = **17.838** | test = **5.516** | gabungan labeled+unlabeled = **18.090**.

OOV rate data uji (**metrik utama: % tipe kata unik test yang tidak ada di kosakata referensi**):

* Terhadap **kosakata labeled saja**: **70.49%** — **3.888 dari 5.516** tipe kata data uji tidak pernah muncul di 300 data latih.
* Terhadap **kosakata labeled + unlabeled (transductif)**: **18.15%** — hanya **1.001 dari 5.516** tipe kata yang tersisa OOV.
* **Penurunan: 52.34 poin persentase (relatif −74.3%).**

Metrik pendukung (berbasis kemunculan token, 40.839 token test):
* OOV token vs labeled saja: **15.68%** (6.404 token OOV) → vs labeled+unlabeled: **2.60%** (1.060 token OOV).

* Justifikasi ilmiah untuk slide/notebook: **7 dari 10 tipe kata** pada ulasan uji asing bagi model yang hanya dilatih pada 300 label; korpus tanpa label memangkasnya menjadi <2 dari 10. Ini alasan numerik mengapa vocabulary TF-IDF **wajib** di-fit transductif pada gabungan labeled+unlabeled (Modul 2, `plans/02`).

## 5. Term Paling Informatif per Kelas (bahan narasi notebook, log-odds ratio, doc-freq ≥ 3)

* **positive (n=175)** — top unigram: `pilihan` (2.03), `bandung` (1.90), `cukup` (1.88), `restoran` (1.79), `batagor` (1.76), `makanan` (1.65); top bigram: `nya enak` (2.39), `makanan nya` (2.17), `makan malam` (1.63), `di sini` (1.63).
  → Pola: **domain kuliner** ("enak", "restoran", "batagor") = sinyal positif dominan.
* **negative (n=94)** — top unigram: `dia` (3.33), `marah` (3.21), `gua` (3.21), `bayar` (2.92), `malah` (2.92), `mengecewakan` (2.92), `pki` (2.92), `islam` (2.92); top bigram: `tidak bisa` (3.33), `tidak enak` (1.82), `saya tidak` (1.82).
  → Pola: campuran **keluhan transaksional** + **opini sosial-politik** (konfirmasi multimodal-topik di Bagian 2 AGENTS.md); negasi `tidak + X` adalah pola bigram kunci → justifikasi n-gram (1,2).
* **neutral (n=31)** — top unigram: `pilkada` (3.52), `2018` (3.52), `demokrat` (2.82), `rp` (2.64), `joko` (2.54), `pilgub` (2.54); top bigram: `hari ini` (1.73), `pada tanggal` (1.73), `terima kasih` (1.44).
  → Pola: **kabar/fakta informatif** (tokoh, tanggal, angka) tanpa muatan evaluatif — kelas ini paling sedikit sampelnya (31) sehingga paling rentan **punah** akibat pseudo-labeling buta.

## 6. Chi-Square Top-15 Unigram & Bigram (fig `04_top_ngrams.png`)

* Top unigram pemisah kelas: `nya` (χ²=84.20), `makanan` (55.87), `enak` (45.46), `tempat` (42.96), `bandung` (33.50), `pilkada` (26.03), `pdip` (26.03).
* Top bigram pemisah kelas: `nya enak` (20.71), `tidak bisa` (17.53), `presiden joko` / `joko widodo` / `anies baswedan` / `di pilgub` (17.35), `di sini` (16.75), `makanan nya` (16.43), `tempat nya` (14.62), `harga nya` (12.51).
* *Implikasi:* fitur **bigram** membawa informasi negasi (`tidak bisa`, `tidak enak`) dan konteks (`harga nya`) yang tidak tercakup unigram → membenarkan konfigurasi `TfidfVectorizer(ngram_range=(1,2))` pada Modul 2.

## 7. Kalimat Kunci Siap Pakai (notebook & slide)

1. "Dari 5.516 tipe kata pada data uji, **3.888 (70.49%) tidak pernah terlihat** oleh model yang hanya mengenal 300 ulasan berlabel; korpus 10.633 ulasan tanpa label menekannya menjadi **1.001 (18.15%)** — pemangkasan **52.34 poin persentase**."
2. "Profil panjang ulasan nyidentik di ketiga dataset (mean **33.45 / 33.05 / 32.78** kata; median **29 / 28 / 28**) — bukti kuat satu distribusi sumber yang membenarkan pendekatan transductif."
3. "Kelas `neutral` hanya **31 sampel (10.33%)** — tanpa mekanisme *class-balanced* pada pseudo-labeling, kelas ini dipastikan menjadi korban pertama *confirmation bias*."

---

**Reproduksi:** `.venv/bin/python src/eda/run_eda.py` (memperbarui `results/eda/eda_stats.json` + 4 figur `figures/` dalam satu run, backend matplotlib `Agg`/headless, ±6 detik).
