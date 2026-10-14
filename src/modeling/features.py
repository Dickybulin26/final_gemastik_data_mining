"""
Ekstraksi fitur untuk klasifikasi teks menggunakan TF-IDF dan SVD.

TF-IDF (Term Frequency - Inverse Document Frequency):
    Mengubah teks menjadi representasi angka berdasarkan seberapa penting
    sebuah kata dalam suatu dokumen dibandingkan seluruh korpus.

SVD (Singular Value Decomposition):
    Teknik untuk mengurangi jumlah fitur (dimensi) agar komputasi lebih cepat
    sambil mempertahankan informasi penting.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf(
    train_texts,
    *,
    max_features: int = 30_000,
    ngram_range: Tuple[int, int] = (1, 2),
    sublinear_tf: bool = True,
    min_df: int = 2,
) -> TfidfVectorizer:
    """Buat dan latih TF-IDF vectorizer pada kumpulan teks.

    Args:
        train_texts: Kumpulan teks untuk membuat kosakata TF-IDF
        max_features: Jumlah maksimum kata/frasa yang diambil (default: 30.000)
        ngram_range: Rentang n-gram, (1,2) artinya unigram + bigram
                     Contoh bigram: "tidak mau", "sangat bagus"
        sublinear_tf: Jika True, gunakan log(1 + tf) agar kata yang sangat
                      sering muncul tidak mendominasi
        min_df: Kata harus muncul minimal di sekian dokumen (filter kata langka)

    Returns:
        TfidfVectorizer yang sudah dilatih (siap dipakai transform)
    """
    vec = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=sublinear_tf,
        min_df=min_df,
    )
    vec.fit(train_texts)
    return vec


def reduce_svd(
    X: sp.spmatrix,
    n_components: int = 200,
    random_state: int = 42,
) -> Tuple[np.ndarray, TruncatedSVD]:
    """Kurangi dimensi data dengan Truncated SVD.

    Berguna ketika jumlah fitur TF-IDF sangat besar (misalnya 50.000)
    dan perlu dikurangi menjadi lebih kecil (misalnya 200) agar
    model bisa berjalan lebih cepat.

    Args:
        X: Matriks sparse (jarang) dari TF-IDF
        n_components: Jumlah dimensi output (default: 200)
        random_state: Seed untuk reprodusibilitas

    Returns:
        Tuple berisi:
        - X_reduced: Matriks padat (dense) yang sudah direduksi
        - svd: Objek SVD (bisa dipakai untuk transform data baru)
    """
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    X_reduced = svd.fit_transform(X)
    return X_reduced, svd
