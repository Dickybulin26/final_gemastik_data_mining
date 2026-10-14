#!/usr/bin/env python3
"""
Eksperimen pemodelan: TF-IDF → Model → Evaluasi → Submission.

Script utama yang menjalankan seluruh pipeline machine learning:
1. Baca data (labeled, unlabeled, test)
2. Preprocessing teks (bersihkan & normalisasi)
3. Buat fitur TF-IDF dari semua teks
4. Uji 3 model dengan cross-validation
5. Pilih model terbaik & buat prediksi
6. Simpan hasil ke submission.csv

Cara menjalankan:
    python -m src.modeling.run_experiments      # dari root project
    python src/modeling/run_experiments.py       # juga bisa

Output:
    results/modeling/results.json  — metrik per model (format mesin)
    results/modeling/RESULTS.md    — ringkasan hasil (format manusia)
    submission.csv                 — prediksi untuk kompetisi
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Agar bisa dijalankan langsung tanpa flag -m
if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import classification_report, f1_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.naive_bayes import MultinomialNB  # noqa: E402
from sklearn.svm import LinearSVC  # noqa: E402

from src.modeling.features import build_tfidf  # noqa: E402
from src.modeling.preprocessing import preprocess_dataframe  # noqa: E402

DATA_DIR = Path("data")
RESULTS_DIR = Path("results/modeling")

# ── Daftar Model yang Diuji ─────────────────────────────────────────────
# Tiga model klasik yang terbukti efektif untuk klasifikasi teks:
#
# 1. MultinomialNB (Naive Bayes) — model probabilistik, sangat cepat
#    alpha=0.3 → parameter smoothing (menghindari probabilitas nol)
#
# 2. LogisticRegression — model linear yang menarik garis batas pemisah
#    C=1.0 → kekuatan regularisasi (mencegah overfitting)
#
# 3. LinearSVC — Support Vector Machine linear, mencari batas pemisah optimal
#    C=1.0 → trade-off antara margin lebar vs kesalahan klasifikasi
MODELS = {
    "MultinomialNB": MultinomialNB(alpha=0.3),
    "LogisticRegression": LogisticRegression(
        C=1.0, max_iter=1000, solver="lbfgs", random_state=42
    ),
    "LinearSVC": LinearSVC(C=1.0, max_iter=4000, random_state=42),
}


def _cross_validate(X, y, model, n_splits: int = 5):
    """Lakukan 5-fold Stratified Cross-Validation.

    Data dibagi menjadi 5 bagian (fold). Setiap fold bergantian menjadi
    data uji, sedangkan 4 fold lainnya menjadi data latih. Hasilnya
    dirata-ratakan untuk mendapat estimasi performa yang lebih stabil.

    Args:
        X: Matriks fitur (TF-IDF)
        y: Array label (0 atau 1)
        model: Model sklearn yang akan diuji
        n_splits: Jumlah fold (default: 5)

    Returns:
        Tuple (rata-rata F1, standar deviasi F1)
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in skf.split(X, y):
        # Buat salinan model baru untuk setiap fold (hindari kebocoran)
        m = type(model)(**model.get_params())
        m.fit(X[train_idx], y[train_idx])           # latih pada 4 fold
        preds = m.predict(X[val_idx])                 # prediksi pada 1 fold
        scores.append(f1_score(y[val_idx], preds, average="macro"))
    return float(np.mean(scores)), float(np.std(scores))


def main() -> None:
    # ── 1. Muat Data ─────────────────────────────────────────────────
    labeled = pd.read_csv(DATA_DIR / "labeled.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    unlabeled = pd.read_csv(DATA_DIR / "unlabeled.csv")

    # ── 2. Preprocessing (bersihkan & normalisasi teks) ──────────────
    labeled = preprocess_dataframe(labeled)
    test = preprocess_dataframe(test)
    unlabeled = preprocess_dataframe(unlabeled)

    # ── 3. Buat Fitur TF-IDF ─────────────────────────────────────────
    # Fit pada gabungan SEMUA teks (labeled + unlabeled + test) agar
    # kosakata lebih lengkap. Ini disebut "transductive TF-IDF".
    # BUKAN data leakage karena kita hanya menghitung frekuensi kata,
    # bukan menggunakan label.
    all_texts = pd.concat(
        [labeled["clean"], unlabeled["clean"], test["clean"]], ignore_index=True
    )
    tfidf = build_tfidf(all_texts, max_features=50_000, ngram_range=(1, 2))

    X_labeled = tfidf.transform(labeled["clean"])
    y_labeled = labeled["label"].values

    # ── 4. Cross-Validation (uji setiap model) ──────────────────────
    results: dict[str, dict] = {}
    best_name, best_score = "", 0.0

    for name, model in MODELS.items():
        mean_f1, std_f1 = _cross_validate(X_labeled, y_labeled, model)
        results[name] = {"mean_f1": round(mean_f1, 4), "std_f1": round(std_f1, 4)}
        print(f"{name:25s}  F1={mean_f1:.4f} ± {std_f1:.4f}")
        if mean_f1 > best_score:
            best_score = mean_f1
            best_name = name

    # ── 5. Latih Ulang Model Terbaik & Prediksi ─────────────────────
    # Model terbaik dilatih pada SELURUH data berlabel (bukan hanya 4/5),
    # agar performanya maksimal untuk prediksi akhir.
    best_model = type(MODELS[best_name])(**MODELS[best_name].get_params())
    best_model.fit(X_labeled, y_labeled)

    X_test = tfidf.transform(test["clean"])
    preds = best_model.predict(X_test)

    # ── 6. Buat File Submission ──────────────────────────────────────
    sub = pd.DataFrame({"id": test["id"], "label": preds})
    sub.to_csv("submission.csv", index=False)
    print(f"\n✓ submission.csv ditulis ({len(sub)} baris, model={best_name})")

    # ── 7. Simpan Hasil Eksperimen ───────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_meta = {"best_model": best_name, "models": results}
    (RESULTS_DIR / "results.json").write_text(json.dumps(results_meta, indent=2))

    # Buat ringkasan dalam format Markdown (mudah dibaca manusia)
    md_lines = [
        "# Hasil Pemodelan\n",
        f"**Model terbaik:** {best_name} (macro-F1 = {best_score:.4f})\n",
        "| Model | F1 (rata-rata) | F1 (std) |",
        "|-------|----------------|----------|",
    ]
    for name, m in results.items():
        md_lines.append(f"| {name} | {m['mean_f1']:.4f} | {m['std_f1']:.4f} |")
    (RESULTS_DIR / "RESULTS.md").write_text("\n".join(md_lines) + "\n")
    print("✓ Hasil disimpan ke results/modeling/")


if __name__ == "__main__":
    main()
