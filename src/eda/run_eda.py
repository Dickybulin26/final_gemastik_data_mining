#!/usr/bin/env python3
"""
Jalankan seluruh pipeline EDA dan hasilkan grafik + statistik JSON.

Script ini menganalisis data dan menghasilkan:
- figures/01_class_distribution.png — distribusi label
- figures/02_word_length_hist.png   — histogram panjang teks
- figures/03_oov_comparison.png     — perbandingan OOV rate
- figures/04_top_ngrams.png         — bigram paling sering
- results/eda/eda_stats.json        — statistik dalam format JSON

Cara menjalankan:
    python -m src.eda.run_eda          # dari root project
    python src/eda/run_eda.py          # juga bisa
"""

from __future__ import annotations

import sys
from pathlib import Path

# Agar bisa dijalankan langsung tanpa flag -m
if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402

from src.eda.utils import (  # noqa: E402
    basic_stats,
    plot_class_distribution,
    plot_oov_comparison,
    plot_top_ngrams,
    plot_word_length_hist,
    save_stats,
)

DATA_DIR = Path("data")
RESULTS_DIR = Path("results/eda")


def main() -> None:
    # ── Muat Data ─────────────────────────────────────────────────────
    labeled = pd.read_csv(DATA_DIR / "labeled.csv")
    unlabeled = pd.read_csv(DATA_DIR / "unlabeled.csv")

    print(f"labeled  : {len(labeled):,} baris")
    print(f"unlabeled: {len(unlabeled):,} baris")

    # ── Hitung Statistik Dasar ────────────────────────────────────────
    stats = basic_stats(labeled)
    print("statistik:", stats)

    # ── Buat Grafik ───────────────────────────────────────────────────
    plot_class_distribution(labeled)           # distribusi label 0 vs 1
    plot_word_length_hist(labeled)             # histogram jumlah kata
    plot_top_ngrams(labeled["text"].dropna().tolist(), n=2)  # top bigrams

    # ── Analisis OOV (Out-of-Vocabulary) ──────────────────────────────
    # Bandingkan kosakata labeled vs unlabeled untuk melihat berapa banyak
    # kata di unlabeled yang tidak ada di labeled (justifikasi transductive TF-IDF)
    from src.eda.utils import _tokenise, oov_rate  # noqa: WPS433

    labeled_vocab = set()
    for doc in labeled["text"].dropna():
        labeled_vocab.update(_tokenise(doc))

    rates = {
        "labeled_self": oov_rate(labeled["text"].dropna(), labeled_vocab),
        "unlabeled_vs_labeled": oov_rate(
            unlabeled["text"].dropna(), labeled_vocab
        ),
    }
    plot_oov_comparison(rates)
    stats["oov_rates"] = rates

    # ── Simpan Hasil ──────────────────────────────────────────────────
    save_stats(stats)
    print("✓ EDA selesai — figures/ dan results/eda/ sudah diperbarui.")


if __name__ == "__main__":
    main()
