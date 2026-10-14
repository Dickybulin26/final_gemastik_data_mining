"""
Pipeline pembersihan dan normalisasi teks Bahasa Indonesia.

Modul ini berisi fungsi-fungsi untuk membersihkan teks mentah dari internet
agar bisa diproses oleh model machine learning. Proses meliputi:
1. Mengubah huruf besar → kecil
2. Menghapus URL, mention (@), hashtag (#), angka, dan tanda baca
3. Normalisasi kata slang (misal: "gw" → "saya", "gak" → "tidak")
4. Merapikan huruf berulang (misal: "baguuus" → "bagus")
"""

from __future__ import annotations

import re
import string
from typing import List

import pandas as pd

# ── Kamus Slang Indonesia ───────────────────────────────────────────────
# Berisi 70+ kata slang/singkatan yang umum dipakai di internet Indonesia.
# Setiap kata slang dipetakan ke bentuk bakunya.
# Contoh: "gak" → "tidak", "gw" → "saya", "bgt" → "sangat"
_SLANG_MAP = {
    "gak": "tidak",
    "ga": "tidak",
    "gk": "tidak",
    "tdk": "tidak",
    "ngga": "tidak",
    "nggak": "tidak",
    "gw": "saya",
    "gue": "saya",
    "gua": "saya",
    "lu": "kamu",
    "lo": "kamu",
    "emg": "memang",
    "emang": "memang",
    "udah": "sudah",
    "udh": "sudah",
    "blm": "belum",
    "belom": "belum",
    "yg": "yang",
    "dgn": "dengan",
    "dg": "dengan",
    "utk": "untuk",
    "krn": "karena",
    "krna": "karena",
    "tp": "tapi",
    "tpi": "tapi",
    "bgt": "banget",
    "bngt": "banget",
    "banget": "sangat",
    "bkn": "bukan",
    "org": "orang",
    "sm": "sama",
    "jg": "juga",
    "jgn": "jangan",
    "dlm": "dalam",
    "lg": "lagi",
    "aj": "saja",
    "aja": "saja",
    "bs": "bisa",
    "gmn": "bagaimana",
    "gimana": "bagaimana",
    "knp": "kenapa",
    "kenapa": "mengapa",
    "spy": "supaya",
    "stlh": "setelah",
    "sblm": "sebelum",
    "dr": "dari",
    "dri": "dari",
    "pd": "pada",
    "trs": "terus",
    "klo": "kalau",
    "kl": "kalau",
    "kalo": "kalau",
    "msh": "masih",
    "hrs": "harus",
    "ttp": "tetap",
    "ttg": "tentang",
    "thd": "terhadap",
    "tsb": "tersebut",
    "dpt": "dapat",
    "sdh": "sudah",
    "blh": "boleh",
    "sgt": "sangat",
    "bnr": "benar",
    "sy": "saya",
    "ak": "aku",
    "mksd": "maksud",
    "cb": "coba",
    "dmn": "dimana",
    "mn": "mana",
    "ni": "ini",
    "tu": "itu",
    "gt": "gitu",
    "gtu": "begitu",
    "gini": "begini",
    "bgtu": "begitu",
    "ntar": "nanti",
    "tar": "nanti",
}

# Pola regex untuk mencocokkan semua kata slang dalam teks.
# Diurutkan dari yang terpanjang agar kata seperti "nggak" cocok sebelum "ga".
_SLANG_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(_SLANG_MAP, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def _replace_slang(text: str) -> str:
    """Ganti semua kata slang dalam teks dengan bentuk bakunya."""
    return _SLANG_PATTERN.sub(lambda m: _SLANG_MAP[m.group(0).lower()], text)


# ── Fungsi Pembersihan Teks ──────────────────────────────────────────────


def clean_text(text: str) -> str:
    """Bersihkan teks: huruf kecil, hapus URL/mention/hashtag/angka/tanda baca.

    Contoh:
        "Cek @admin http://t.co/abc #gemastik 123!"
        → "cek"
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()                                      # huruf kecil semua
    text = re.sub(r"http\S+|www\.\S+", " ", text)           # hapus URL
    text = re.sub(r"@\w+", " ", text)                        # hapus mention
    text = re.sub(r"#\w+", " ", text)                        # hapus hashtag
    text = re.sub(r"\d+", " ", text)                         # hapus angka
    text = text.translate(str.maketrans("", "", string.punctuation))  # hapus tanda baca
    text = re.sub(r"\s+", " ", text).strip()                 # rapikan spasi
    return text


def normalise(text: str) -> str:
    """Normalisasi teks: ganti slang + rapikan huruf berulang.

    Contoh:
        "gw gak mau baguuuus" → "saya tidak mau bagus"
    """
    text = _replace_slang(text)                              # ganti slang ke bentuk baku
    text = re.sub(r"(.)\1{2,}", r"\1", text)                 # "baguuus" → "bagus"
    return text


def preprocess_series(series: pd.Series) -> pd.Series:
    """Jalankan pipeline lengkap (clean → normalise) pada kolom teks."""
    return series.apply(clean_text).apply(normalise)


def preprocess_dataframe(
    df: pd.DataFrame,
    text_col: str = "text",
    out_col: str = "clean",
) -> pd.DataFrame:
    """Tambahkan kolom 'clean' berisi teks yang sudah dibersihkan.

    Args:
        df: DataFrame yang memiliki kolom teks
        text_col: Nama kolom berisi teks mentah (default: "text")
        out_col: Nama kolom output (default: "clean")

    Returns:
        DataFrame baru dengan kolom tambahan berisi teks bersih
    """
    df = df.copy()
    df[out_col] = preprocess_series(df[text_col])
    return df
