# G3 Semi-supervised: Sentimen dengan Label Terbatas

Tema: inovasi yang hemat biaya. SDG 8 dan 9.

## Latar
Sebuah platform ulasan hanya mampu melabeli 300 ulasan, padahal masih ada puluhan ribu ulasan lain. Mereka ingin tahu apakah data tanpa label membantu atau justru merugikan.

## Data
Folder `data/`:
- `labeled.csv`: id, text, label (300). Label: positive, neutral, negative.
- `unlabeled.csv`: id, text (±10.600).
- `test.csv`: id, text (±1.250). Labelnya disimpan penguji.
- `submission_template.csv`: contoh format jawaban.

Teks berupa ulasan berbahasa Indonesia. Teks berasal dari media sosial dan ulasan asli, sehingga dapat memuat bahasa kasar atau topik sensitif (politik, SARA).

## Tugas
Bangun pengklasifikasi sentimen yang memakai data tanpa label sebaik mungkin. Prediksi `test.csv`, simpan sebagai `submission.csv` (kolom id, label), cek dengan `python check_submission.py submission.csv`. Metrik penilaian: macro-F1.

## Objective
1. Periksa data berlabel dan jelaskan implikasinya terhadap pilihan metrik dan metode.
2. Buat baseline yang hanya memakai 300 label. Tentukan strategi validasi sendiri.
3. Terapkan minimal satu metode semi-supervised (self-training, pseudo-labeling, label spreading, dll.) dan jelaskan parameter pentingnya.
4. Bandingkan dengan baseline pada validasi yang sama. Periksa kualitas pseudo-label bila bisa.
5. Buat `submission.csv` dari pendekatan yang menurut bukti Anda terbaik.
6. Simpulkan jujur: membantu atau tidak, dan kenapa.

## Output
- `submission.csv` yang lolos pemeriksa.
- Tabel baseline vs semi-supervised (macro-F1).
- Analisis pseudo-label: jumlah dan dugaan kualitas.
- Kesimpulan maksimal 5 kalimat di akhir notebook.
- 5 slide: masalah dan SDG, temuan data, metode, hasil dan keterbatasan, rekomendasi.

## Aturan
- 60 menit. Presentasi 5 menit, tanya jawab 5 menit.
- Boleh pakai agentic AI dan internet. Layar dibagikan.
- Satu notebook. Semua klaim harus bisa ditunjukkan dari notebook.
- Hasil negatif yang jujur tetap dihargai.
- Dilarang mencari atau memakai jawaban data uji dari sumber mana pun.

## Penilaian
| Komponen | Poin |
|---|---|
| Macro-F1 di data uji tersembunyi | 30 |
| Metodologi, validasi, verifikasi output AI | 20 |
| Inovasi | 15 |
| Presentasi (jelas dan lengkap) | 25 |
| Kesesuaian tema dan dampak | 10 |

## Pertanyaan juri
- Mengapa Anda memakai metrik validasi itu?
- Apa risiko metode semi-supervised yang Anda pilih dan bagaimana Anda memeriksanya?
- Kalau ada anggaran 300 label lagi, bagaimana Anda memilih datanya?
