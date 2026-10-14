#!/usr/bin/env python3
"""Pemeriksaan cepat (sanity check) untuk submission.csv.

Script ini memvalidasi bahwa file submission.csv memiliki format yang benar
sesuai dengan template yang disediakan kompetisi:
- Kolom harus sama (id, label)
- Jumlah baris harus sama (400)
- Nilai id harus cocok dengan template
- Tidak boleh ada label yang kosong (NaN)

Cara menjalankan:
    python check_submission.py
"""

from pathlib import Path

import pandas as pd

TEMPLATE = Path("data/submission_template.csv")
SUBMISSION = Path("submission.csv")


def main() -> None:
    tpl = pd.read_csv(TEMPLATE)
    sub = pd.read_csv(SUBMISSION)

    errors: list[str] = []

    # Periksa: apakah kolom-kolomnya sama?
    if list(sub.columns) != list(tpl.columns):
        errors.append(f"Kolom tidak cocok: seharusnya {list(tpl.columns)}, dapat {list(sub.columns)}")

    # Periksa: apakah jumlah baris sama?
    if len(sub) != len(tpl):
        errors.append(f"Jumlah baris tidak cocok: seharusnya {len(tpl)}, dapat {len(sub)}")

    # Periksa: apakah id-nya identik?
    if not sub["id"].equals(tpl["id"]):
        errors.append("Nilai kolom id berbeda dari template")

    # Periksa: apakah ada label yang kosong?
    if sub["label"].isna().any():
        errors.append(f"Ditemukan {sub['label'].isna().sum()} label kosong (NaN)")

    if errors:
        print("❌ GAGAL")
        for e in errors:
            print(f"   • {e}")
        raise SystemExit(1)

    print(f"✅ submission.csv valid — {len(sub)} baris, "
          f"label: {sorted(sub['label'].unique())}")


if __name__ == "__main__":
    main()
