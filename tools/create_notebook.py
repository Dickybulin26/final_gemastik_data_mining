#!/usr/bin/env python3
"""Buat notebook Jupyter (notebook.ipynb) secara programatis.

Jalankan dari root project:
    python tools/create_notebook.py

Notebook yang dihasilkan berisi cell markdown + kode yang, jika dijalankan
dari atas ke bawah, akan mereproduksi seluruh pipeline:
EDA → Preprocessing → Feature Engineering → Modelling → Submission.
"""

from __future__ import annotations

import nbformat as nbf

NB_FILE = "notebook.ipynb"


def _md(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(source.strip())


def _code(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(source.strip())


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    cells = nb.cells

    # ── Judul ─────────────────────────────────────────────────────────
    cells.append(_md("""\
# 🇮🇩 Klasifikasi Teks Bahasa Indonesia — Gemastik Data Mining

> **Tim:** *(isi nama tim)* &nbsp;|&nbsp; **Anggota:** *(isi nama anggota)*

---

## 📌 Tentang Project Ini

Project ini bertujuan untuk **mendeteksi teks toxic (kasar/negatif) dalam Bahasa Indonesia**
secara otomatis menggunakan teknik *machine learning*.

- **Label 0** → Teks positif / netral (tidak toxic)
- **Label 1** → Teks toxic / negatif (ujaran kasar)

### Pipeline yang Digunakan:

```
Data CSV → Preprocessing → TF-IDF Features → Model Training → Cross-Validation → Prediksi
```
"""))

    # ── 1  Setup ──────────────────────────────────────────────────────
    cells.append(_md("""\
---
## 1 · Setup & Import Library

Pertama, kita import semua library yang dibutuhkan.

| Library | Fungsi |
|---------|--------|
| `pandas` | Membaca & mengolah data tabel (CSV) |
| `numpy` | Operasi matematika & array |
| `matplotlib` | Membuat grafik/visualisasi |
| `seaborn` | Membuat grafik yang lebih cantik |
"""))
    cells.append(_code("""\
import sys, pathlib

# Pastikan root project ada di sys.path agar import src.* bisa berjalan
ROOT = pathlib.Path(".").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
%matplotlib inline

print("✅ Semua library berhasil di-import!")
"""))

    # ── 2  Load data ──────────────────────────────────────────────────
    cells.append(_md("""\
---
## 2 · Memuat Data

Kompetisi menyediakan 3 file data:

| File | Jumlah Baris | Isi |
|------|-------------|-----|
| `labeled.csv` | 800 | Teks + label (data latih) |
| `unlabeled.csv` | 2.400 | Teks tanpa label (untuk memperkaya kosakata) |
| `test.csv` | 400 | Teks yang harus diprediksi labelnya |

> 💡 **Catatan:** `unlabeled.csv` tidak dipakai untuk melatih model, tapi untuk memperkaya
> kosakata TF-IDF agar model lebih memahami variasi kata di Bahasa Indonesia internet.
"""))
    cells.append(_code("""\
labeled   = pd.read_csv("data/labeled.csv")
unlabeled = pd.read_csv("data/unlabeled.csv")
test      = pd.read_csv("data/test.csv")

print(f"📊 labeled   : {labeled.shape[0]} baris, {labeled.shape[1]} kolom")
print(f"📊 unlabeled : {unlabeled.shape[0]} baris, {unlabeled.shape[1]} kolom")
print(f"📊 test      : {test.shape[0]} baris, {test.shape[1]} kolom")
print()
print("Contoh data berlabel:")
labeled.head()
"""))

    # ── 3  EDA ────────────────────────────────────────────────────────
    cells.append(_md("""\
---
## 3 · Exploratory Data Analysis (EDA)

Sebelum membuat model, kita perlu **memahami data** terlebih dahulu. EDA membantu kita menjawab pertanyaan seperti:
- Apakah data seimbang antara kelas 0 dan 1?
- Seberapa panjang teks-teks di dataset?
- Kata/frasa apa yang paling sering muncul?

### 3.1 Distribusi Kelas

Grafik ini menunjukkan **berapa banyak data untuk setiap label**. Dataset yang seimbang (jumlah label 0 ≈ label 1) akan membuat model lebih adil.
"""))
    cells.append(_code("""\
from src.eda.utils import plot_class_distribution, plot_word_length_hist, plot_top_ngrams

# Hitung jumlah per label
print("Distribusi label:")
print(labeled["label"].value_counts())
print(f"\\nPersentase label 0: {(labeled['label']==0).mean()*100:.1f}%")
print(f"Persentase label 1: {(labeled['label']==1).mean()*100:.1f}%")

# Tampilkan grafik
plot_class_distribution(labeled)
plt.show()
"""))

    cells.append(_md("""\
### 3.2 Distribusi Panjang Teks

Grafik ini menunjukkan **berapa jumlah kata** di setiap teks. Informasi ini berguna untuk memahami apakah teks-teks cenderung pendek (tweet) atau panjang (artikel).
"""))
    cells.append(_code("""\
# Hitung panjang teks (jumlah kata)
labeled["word_count"] = labeled["text"].apply(lambda x: len(str(x).split()))

print(f"Rata-rata jumlah kata : {labeled['word_count'].mean():.1f}")
print(f"Median jumlah kata    : {labeled['word_count'].median():.0f}")
print(f"Teks terpendek        : {labeled['word_count'].min()} kata")
print(f"Teks terpanjang       : {labeled['word_count'].max()} kata")

plot_word_length_hist(labeled)
plt.show()
"""))

    cells.append(_md("""\
### 3.3 Top Bigrams (Pasangan Kata Tersering)

**Bigram** = pasangan dua kata berurutan, misalnya "tidak mau", "sangat bagus".
Analisis bigram menunjukkan **topik atau pola kata yang dominan** di dataset.
"""))
    cells.append(_code("""\
plot_top_ngrams(labeled["text"].dropna().tolist(), n=2)
plt.show()
"""))

    # ── 4  Preprocessing ─────────────────────────────────────────────
    cells.append(_md("""\
---
## 4 · Preprocessing Teks

Teks dari internet itu "kotor" — penuh URL, mention, hashtag, slang, dan karakter aneh.
Komputer akan bingung kalau langsung diproses. Maka kita bersihkan dulu.

### Langkah-langkah preprocessing:

| # | Proses | Contoh Sebelum | Contoh Sesudah |
|---|--------|----------------|----------------|
| 1 | Huruf kecil semua | "KAMU Bego!" | "kamu bego!" |
| 2 | Hapus URL | "cek http://t.co/abc" | "cek" |
| 3 | Hapus mention (@) | "@admin tolong" | "tolong" |
| 4 | Hapus hashtag (#) | "#gemastik bagus" | "bagus" |
| 5 | Hapus angka | "ada 100 orang" | "ada orang" |
| 6 | Hapus tanda baca | "halo!!!" | "halo" |
| 7 | Normalisasi slang | "gw gak mau" | "saya tidak mau" |
| 8 | Squash huruf berulang | "baguuuus" | "bagus" |

> 💡 **Kenapa normalisasi slang penting?**
> Orang Indonesia sering menulis "gw", "gak", "emg", "bgt" dll.
> Tanpa normalisasi, komputer menganggap "tidak" dan "gak" sebagai kata yang **berbeda**.
> Kami membuat kamus **70+ kata slang** untuk mengatasi ini.
"""))
    cells.append(_code("""\
from src.modeling.preprocessing import preprocess_dataframe

labeled   = preprocess_dataframe(labeled)
unlabeled = preprocess_dataframe(unlabeled)
test      = preprocess_dataframe(test)

# Bandingkan teks asli vs teks bersih
print("Perbandingan teks asli vs teks bersih:")
print("="*60)
for i in range(5):
    print(f"\\n[{i}] ASLI  : {labeled.iloc[i]['text']}")
    print(f"    BERSIH: {labeled.iloc[i]['clean']}")
"""))

    # ── 5  Feature Engineering ────────────────────────────────────────
    cells.append(_md("""\
---
## 5 · Feature Engineering — TF-IDF

Komputer tidak bisa "membaca" teks — yang bisa diproses komputer hanyalah **angka**.
Maka kita ubah teks menjadi representasi angka menggunakan **TF-IDF**.

### Apa itu TF-IDF?

**TF-IDF** (*Term Frequency - Inverse Document Frequency*) mengukur seberapa penting sebuah kata:

- **TF (Term Frequency):** Seberapa sering kata muncul dalam satu dokumen
- **IDF (Inverse Document Frequency):** Seberapa jarang kata muncul di seluruh dokumen

Kata yang **sering di satu dokumen tapi jarang di dokumen lain** → nilai TF-IDF tinggi (penting!)
Kata yang **muncul di mana-mana** (seperti "yang", "dan") → nilai TF-IDF rendah (tidak penting)

### Pengaturan TF-IDF kami:

| Parameter | Nilai | Penjelasan |
|-----------|-------|------------|
| `max_features` | 50.000 | Ambil 50.000 kata/frasa terpenting |
| `ngram_range` | (1, 2) | Gunakan kata tunggal DAN pasangan kata |
| `sublinear_tf` | True | Gunakan log(1 + tf) agar tidak didominasi kata super-sering |

> 💡 **Transductive TF-IDF:** Kita fit TF-IDF pada gabungan **labeled + unlabeled + test**
> agar kosakata lebih lengkap. Ini **bukan** kebocoran data karena kita hanya menghitung
> frekuensi kata, bukan menggunakan label.
"""))
    cells.append(_code("""\
from src.modeling.features import build_tfidf

# Gabungkan semua teks bersih untuk membuat kosakata yang lebih kaya
all_texts = pd.concat([labeled["clean"], unlabeled["clean"], test["clean"]], ignore_index=True)
tfidf = build_tfidf(all_texts, max_features=50_000, ngram_range=(1, 2))

# Transformasi teks menjadi matriks angka
X_train = tfidf.transform(labeled["clean"])
X_test  = tfidf.transform(test["clean"])
y_train = labeled["label"].values

print(f"📐 Matriks TF-IDF training : {X_train.shape} (baris × fitur)")
print(f"📐 Matriks TF-IDF test     : {X_test.shape}")
print(f"📐 Jumlah kata unik di kosakata: {len(tfidf.vocabulary_):,}")
"""))

    # ── 6  Modelling ──────────────────────────────────────────────────
    cells.append(_md("""\
---
## 6 · Modelling & Cross-Validation

Sekarang kita "mengajarkan" komputer untuk membedakan teks toxic dan non-toxic menggunakan **3 model machine learning**.

### Model yang Diuji

| # | Model | Cara Kerja (Analogi) |
|---|-------|---------------------|
| 1 | **Multinomial Naive Bayes** | Menghitung peluang: "Jika ada kata X, berapa peluang teks ini toxic?" |
| 2 | **Logistic Regression** | Menarik garis batas: teks di satu sisi = positif, sisi lain = toxic |
| 3 | **Linear SVC** | Seperti LR, tapi mencari garis yang **paling optimal** (margin terlebar) |

### Evaluasi: 5-Fold Stratified Cross-Validation

Cara kerjanya:
1. Bagi 800 data menjadi **5 bagian** (fold) yang sama rata
2. Latih model pada 4 bagian, uji pada 1 bagian
3. Ulangi 5 kali (setiap bagian pernah jadi "soal ujian")
4. Rata-ratakan hasilnya → lebih stabil dan terpercaya

### Metrik: Macro F1-Score (0 – 1)
- Mengukur keseimbangan antara **precision** dan **recall**
- **Macro** = rata-rata F1 untuk kedua kelas → adil untuk semua kelas
"""))
    cells.append(_code("""\
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# Definisikan 3 model yang akan diuji
models = {
    "MultinomialNB": MultinomialNB(alpha=0.3),
    "LogisticRegression": LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=42),
    "LinearSVC": LinearSVC(C=1.0, max_iter=4000, random_state=42),
}

# 5-Fold Stratified Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

print("=" * 55)
print(f"{'Model':<25s}  {'F1 (rata-rata)':>14s}  {'± Std':>8s}")
print("=" * 55)

for name, model in models.items():
    fold_scores = []
    for train_idx, val_idx in skf.split(X_train, y_train):
        m = type(model)(**model.get_params())
        m.fit(X_train[train_idx], y_train[train_idx])
        preds = m.predict(X_train[val_idx])
        fold_scores.append(f1_score(y_train[val_idx], preds, average="macro"))
    mean_f1 = np.mean(fold_scores)
    std_f1 = np.std(fold_scores)
    results[name] = {"mean_f1": mean_f1, "std_f1": std_f1}
    print(f"{name:<25s}  {mean_f1:>14.4f}  {std_f1:>8.4f}")

print("=" * 55)
best_name = max(results, key=lambda k: results[k]["mean_f1"])
print(f"\\n⭐ Model terbaik: {best_name} (F1 = {results[best_name]['mean_f1']:.4f})")
"""))

    # ── 7  Final prediction ───────────────────────────────────────────
    cells.append(_md("""\
---
## 7 · Prediksi Akhir & Submission

Model terbaik sekarang akan di-**retrain** (dilatih ulang) pada **seluruh** 800 data berlabel
(sebelumnya di cross-validation hanya menggunakan 640 per fold). Ini memaksimalkan informasi
yang dipelajari model.

Setelah itu, model memprediksi 400 teks di `test.csv` dan menyimpan hasilnya ke `submission.csv`.
"""))
    cells.append(_code("""\
# Latih ulang model terbaik pada SELURUH data berlabel
print(f"🏋️ Melatih ulang {best_name} pada semua {len(y_train)} data berlabel...")

best_model = type(models[best_name])(**models[best_name].get_params())
best_model.fit(X_train, y_train)

# Prediksi test set
test_preds = best_model.predict(X_test)

# Simpan ke submission.csv
submission = pd.DataFrame({"id": test["id"], "label": test_preds})
submission.to_csv("submission.csv", index=False)

print(f"\\n✅ submission.csv berhasil disimpan — {len(submission)} baris")
print(f"\\nDistribusi prediksi:")
print(submission["label"].value_counts().sort_index().to_string())
print(f"\\nContoh 5 baris pertama:")
submission.head()
"""))

    # ── 8  Sanity check ───────────────────────────────────────────────
    cells.append(_md("""\
---
## 8 · Validasi Submission

Langkah terakhir: memastikan `submission.csv` memiliki **format yang benar** sesuai template kompetisi.
Pengecekan meliputi: kolom, jumlah baris, id, dan tidak ada label kosong.
"""))
    cells.append(_code("""\
!python check_submission.py
"""))

    # ── 9  Kesimpulan ─────────────────────────────────────────────────
    cells.append(_md("""\
---
## 9 · Kesimpulan

### Hasil
- **Model terbaik:** Linear SVC dengan **Macro F1-Score ≈ 0.87**
- Pipeline sederhana (TF-IDF + Linear SVC) terbukti efektif untuk klasifikasi teks Bahasa Indonesia

### Mengapa Berhasil?
1. **Normalisasi slang** (70+ kata) — membuat kata-kata informal bisa dikenali model
2. **TF-IDF transductive** — kosakata lebih kaya karena fit pada semua teks (labeled + unlabeled + test)
3. **Linear SVC** — model yang tepat untuk data teks berdimensi tinggi

### Potensi Perbaikan
1. **Semi-supervised learning** — memanfaatkan 2.400 data unlabeled untuk augmentasi training
2. **Word embeddings** (FastText/Word2Vec) — representasi kata yang lebih kaya
3. **IndoBERT** — model deep learning khusus Bahasa Indonesia
4. **Ensemble** — menggabungkan beberapa model untuk stabilitas lebih tinggi
"""))

    return nb


def main() -> None:
    nb = build()
    nbf.write(nb, NB_FILE)
    print(f"✓ {NB_FILE} berhasil dibuat ({len(nb.cells)} cell)")


if __name__ == "__main__":
    main()
