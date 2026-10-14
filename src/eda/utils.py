"""
Fungsi-fungsi bantu untuk Exploratory Data Analysis (EDA).

Modul ini menyediakan:
- Penghitungan statistik dasar dataset (jumlah baris, distribusi label, dll)
- Pembuatan grafik/visualisasi (distribusi kelas, histogram panjang teks, n-gram)
- Analisis Out-of-Vocabulary (OOV) — seberapa banyak kata yang tidak dikenal
- Penyimpanan hasil ke file JSON
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib
matplotlib.use("Agg")  # backend non-interaktif (tidak perlu GUI)
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


# ── Fungsi Bantu Kecil ───────────────────────────────────────────────────


def _safe_len(x: Any) -> int:
    """Hitung jumlah kata; kembalikan 0 jika bukan string."""
    if isinstance(x, str):
        return len(x.split())
    return 0


def basic_stats(df: pd.DataFrame, text_col: str = "text") -> Dict[str, Any]:
    """Hitung statistik dasar dataset.

    Mengembalikan dictionary berisi:
    - n_rows: jumlah total baris
    - label_counts: jumlah per label (jika ada kolom 'label')
    - avg_words: rata-rata jumlah kata per dokumen
    - median_words: median jumlah kata per dokumen
    """
    stats: Dict[str, Any] = {"n_rows": len(df)}
    if "label" in df.columns:
        stats["label_counts"] = df["label"].value_counts().to_dict()
    if text_col in df.columns:
        lengths = df[text_col].apply(_safe_len)
        stats["avg_words"] = round(float(lengths.mean()), 1)
        stats["median_words"] = int(lengths.median())
    return stats


# ── Fungsi Pembuatan Grafik ──────────────────────────────────────────────


def plot_class_distribution(
    df: pd.DataFrame, out_path: str | Path = "figures/01_class_distribution.png"
) -> Path:
    """Buat grafik batang distribusi label (berapa banyak data per kelas).

    Grafik ini menunjukkan apakah dataset seimbang atau tidak.
    Dataset seimbang = jumlah label 0 dan 1 kira-kira sama.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    counts = df["label"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 3))
    counts.plot.bar(ax=ax, color=["#4c72b0", "#55a868"])
    ax.set_title("Class Distribution")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_word_length_hist(
    df: pd.DataFrame,
    text_col: str = "text",
    out_path: str | Path = "figures/02_word_length_hist.png",
) -> Path:
    """Buat histogram distribusi jumlah kata per dokumen.

    Grafik ini menunjukkan apakah teks-teks di dataset cenderung
    pendek atau panjang. Berguna untuk menentukan strategi fitur.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lengths = df[text_col].apply(_safe_len)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(lengths, bins=40, edgecolor="white")
    ax.set_title("Word-count distribution")
    ax.set_xlabel("Words")
    ax.set_ylabel("Documents")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_top_ngrams(
    corpus: Sequence[str],
    n: int = 2,
    top_k: int = 15,
    out_path: str | Path = "figures/04_top_ngrams.png",
) -> Path:
    """Buat grafik batang horizontal dari n-gram paling sering muncul.

    N-gram = pasangan kata berurutan. Bigram (n=2) contoh: "tidak mau".
    Grafik ini menunjukkan topik atau pola kata yang dominan di dataset.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    counter: Counter[tuple[str, ...]] = Counter()
    for doc in corpus:
        if not isinstance(doc, str):
            continue
        tokens = doc.lower().split()
        counter.update(zip(*(tokens[i:] for i in range(n))))
    most = counter.most_common(top_k)
    labels = [" ".join(ng) for ng, _ in most]
    values = [c for _, c in most]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(labels[::-1], values[::-1])
    ax.set_title(f"Top-{top_k} {n}-grams")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ── Analisis Out-of-Vocabulary (OOV) ────────────────────────────────────
# OOV = kata-kata yang ada di satu dataset tapi tidak ada di dataset lain.
# OOV rate tinggi berarti banyak kata "asing" → justifikasi untuk
# menggunakan transductive TF-IDF (fit pada semua teks).


def _tokenise(text: str) -> list[str]:
    """Pecah teks menjadi daftar kata (token) sederhana."""
    return re.findall(r"\b\w+\b", text.lower())


def oov_rate(corpus: Sequence[str], vocab: set[str]) -> float:
    """Hitung persentase kata yang tidak ada di vocab (Out-of-Vocabulary).

    Args:
        corpus: Kumpulan teks yang akan diperiksa
        vocab: Kumpulan kata yang dikenal (kosakata referensi)

    Returns:
        Nilai 0.0–1.0 (0% – 100% kata tidak dikenal)
    """
    total = oov = 0
    for doc in corpus:
        if not isinstance(doc, str):
            continue
        for tok in _tokenise(doc):
            total += 1
            if tok not in vocab:
                oov += 1
    return round(oov / total, 4) if total else 0.0


def plot_oov_comparison(
    rates: Dict[str, float],
    out_path: str | Path = "figures/03_oov_comparison.png",
) -> Path:
    """Buat grafik perbandingan OOV rate antar sumber data."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(rates.keys(), rates.values(), color="#dd8452")
    ax.set_ylabel("OOV rate")
    ax.set_title("Out-of-Vocabulary rate by source")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ── Penyimpanan Hasil ────────────────────────────────────────────────────


def save_stats(stats: Dict[str, Any], path: str | Path = "results/eda/eda_stats.json") -> Path:
    """Simpan statistik ke file JSON agar bisa dibaca kembali nanti."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    return p
