# 02. Tahapan Implementasi Teknis & Spesifikasi Kode

> **Petunjuk Pelaksana:** Dokumen ini memuat blueprint teknis terperinci, struktur fungsi, parameter hiper, dan tata cara eksekusi modular. Junior programmer atau AI implementer dapat menyalin dan mengadaptasi kode ini secara langsung tanpa keraguan.

---

## Modul 1: Environment & Loading Data

### 1.1 Inisialisasi Environment
Pastikan menggunakan interpreter virtual environment lokal:
```bash
.venv/bin/python
```

### 1.2 Pemuatan Dataset & Verifikasi Skema
```python
import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')

# 1. Pemuatan Data
train_df = pd.read_csv('data/labeled.csv')          # 300 baris (id, text, label)
unlabeled_df = pd.read_csv('data/unlabeled.csv')    # 10.633 baris (id, text)
test_df = pd.read_csv('data/test.csv')              # 1.246 baris (id, text)

# Verifikasi tidak ada null pada kolom teks
assert train_df['text'].isnull().sum() == 0, "Ada nilai null di labeled text"
assert unlabeled_df['text'].isnull().sum() == 0, "Ada nilai null di unlabeled text"
assert test_df['text'].isnull().sum() == 0, "Ada nilai null di test text"
```

### 1.3 Fungsi Preprocessing Teks Bahasa Indonesia
Fungsi pembersihan teks harus cerdas: menormalkan karakter tanpa menghapus kata negasi atau tanda sentimen penting:
```python
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Hapus URL
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    # Hapus user mentions (@user) atau hashtag (#topic)
    text = re.sub(r'@\w+|#\w+', ' ', text)
    # Normalkan karakter berulang (misal: 'bangeeeet' -> 'banget', 'haaaai' -> 'hai')
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    # Ganti karakter non-alfanumerik kecuali spasi
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    # Standarisasi spasi berlebih
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Terapkan pembersihan
train_df['clean_text'] = train_df['text'].apply(clean_text)
unlabeled_df['clean_text'] = unlabeled_df['text'].apply(clean_text)
test_df['clean_text'] = test_df['text'].apply(clean_text)
```

---

## Modul 2: Representasi Fitur & Transductive Vocabulary Fitting

Gunakan **FeatureUnion** menggabungkan Word n-gram dan Character n-gram:
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

def build_vectorizer():
    word_vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True
    )
    char_vec = TfidfVectorizer(
        ngram_range=(3, 5),
        analyzer='char',
        min_df=3,
        sublinear_tf=True
    )
    union_vec = FeatureUnion([
        ('word', word_vec),
        ('char', char_vec)
    ])
    return union_vec

# KUNCI TRANSDUCTIVE: Kosakata di-fit pada seluruh korpus teks (Labeled + Unlabeled)
# Ini memangkas OOV Rate dari 70.49% menjadi 18.15%!
full_corpus = pd.concat([train_df['clean_text'], unlabeled_df['clean_text']], ignore_index=True)
global_vectorizer = build_vectorizer()
global_vectorizer.fit(full_corpus)

# Transformasi fitur
X_labeled_full = global_vectorizer.transform(train_df['clean_text'])
X_unlabeled_full = global_vectorizer.transform(unlabeled_df['clean_text'])
X_test_full = global_vectorizer.transform(test_df['clean_text'])
y_labeled_full = train_df['label'].values
```

---

## Modul 3: Protokol Validasi Stratified 5-Fold CV

Definisikan skema validasi yang identik dan baku:
```python
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report

SKF = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def evaluate_cv_macro_f1(model_factory, X, y, skf=SKF):
    """
    Mengevaluasi model pada 5-Fold Stratified CV dan mengembalikan skor Macro-F1 per fold.
    """
    scores = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_val = X[val_idx], y[val_idx]
        
        clf = model_factory()
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_va)
        f1 = f1_score(y_val, preds, average='macro')
        scores.append(f1)
    return scores
```

---

## Modul 4: Supervised Baseline Modeling (Fase 2)

Latih model-model pembanding dasar pada 300 data berlabel:
```python
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import ComplementNB
from sklearn.ensemble import VotingClassifier

# 1. Model Baseline Factories
def get_lr():
    return LogisticRegression(class_weight='balanced', C=1.0, max_iter=1000, random_state=42)

def get_svm():
    base_svc = LinearSVC(class_weight='balanced', C=0.5, random_state=42)
    return CalibratedClassifierCV(base_svc, cv=3)

def get_cnb():
    return ComplementNB(norm=True)

def get_ensemble():
    return VotingClassifier(
        estimators=[
            ('lr', get_lr()),
            ('svm', get_svm()),
            ('cnb', get_cnb())
        ],
        voting='soft'
    )

# 2. Eksekusi Baseline 5-Fold CV
baseline_models = {
    "Logistic Regression (Balanced)": get_lr,
    "Calibrated LinearSVC (Balanced)": get_svm,
    "Complement Naive Bayes": get_cnb,
    "Soft Voting Ensemble": get_ensemble
}

results = {}
for name, factory in baseline_models.items():
    fold_scores = evaluate_cv_macro_f1(factory, X_labeled_full, y_labeled_full)
    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)
    results[name] = (fold_scores, mean_score, std_score)
    print(f"{name:35s}: Mean Macro-F1 = {mean_score:.4f} (±{std_score:.4f}) | Folds: {[round(s, 4) for s in fold_scores]}")
```

*Catat hasil angka ke dalam [progress-tracker.md](file:///home/dicky/Downloads/G3_Semi_supervised/progress-tracker.md)!*

---

## Modul 5: Implementasi Semi-Supervised Class-Balanced Pseudo-Labeling (CB-PL)

### 5.1 Algoritma Class-Balanced Pseudo-Labeling
```python
from scipy.sparse import vstack

def run_semi_supervised_cv(base_model_factory, X_lab, y_lab, X_unlab, skf=SKF, 
                           top_k_per_class=60, confidence_threshold=0.85, 
                           pseudo_weight=0.5, iterations=1):
    """
    Evaluasi Semi-Supervised CV tanpa kebocoran data (Data Leakage Free):
    Pada setiap fold:
      1. Model di-fit pada X_tr berlabel.
      2. Memprediksi probabilitas pada X_unlab.
      3. Memilih top-K ulasan berkeyakinan tertinggi untuk SETIAP kelas secara proporsional.
      4. Menggabungkan train_fold + pseudo_labeled dengan sample weights.
      5. Melatih ulang model dan mengevaluasi HANYA pada X_va (validation fold murni).
    """
    cv_scores = []
    classes = np.unique(y_lab)
    diagnostic_info = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_lab, y_lab)):
        X_tr = X_lab[train_idx]
        y_tr = y_lab[train_idx]
        X_va = X_lab[val_idx]
        y_va = y_lab[val_idx]
        
        # Bobot data asli = 1.0
        sample_weights_tr = np.ones(X_tr.shape[0])
        current_X_tr = X_tr.copy()
        current_y_tr = y_tr.copy()
        
        for it in range(iterations):
            clf = base_model_factory()
            if hasattr(clf, "fit") and "sample_weight" in clf.fit.__code__.co_varnames:
                clf.fit(current_X_tr, current_y_tr, sample_weight=sample_weights_tr)
            else:
                clf.fit(current_X_tr, current_y_tr)
                
            # Prediksi probabilitas pada unlabeled pool
            probs = clf.predict_proba(X_unlab)
            max_probs = np.max(probs, axis=1)
            pred_classes = classes[np.argmax(probs, axis=1)]
            
            # Seleksi berimbang per kelas (Class-Balanced Selection)
            selected_indices = []
            selected_labels = []
            selected_weights = []
            
            for c in classes:
                c_mask = (pred_classes == c) & (max_probs >= confidence_threshold)
                c_indices = np.where(c_mask)[0]
                if len(c_indices) > 0:
                    # Urutkan berdasarkan probabilitas tertinggi
                    sorted_c_idx = c_indices[np.argsort(-max_probs[c_indices])]
                    # Ambil top K
                    chosen = sorted_c_idx[:top_k_per_class]
                    selected_indices.extend(chosen)
                    selected_labels.extend([c] * len(chosen))
                    selected_weights.extend([pseudo_weight] * len(chosen))
            
            if len(selected_indices) == 0:
                break
                
            # Catat diagnostik hanya pada fold 0 untuk analisis
            if fold == 0 and it == 0:
                counts = pd.Series(selected_labels).value_counts().to_dict()
                diagnostic_info.append({
                    "total_pseudo": len(selected_labels),
                    "distribution": counts,
                    "avg_confidence": np.mean(max_probs[selected_indices])
                })
                
            # Tambahkan ke training pool
            X_pseudo = X_unlab[selected_indices]
            y_pseudo = np.array(selected_labels)
            w_pseudo = np.array(selected_weights)
            
            current_X_tr = vstack([current_X_tr, X_pseudo])
            current_y_tr = np.concatenate([current_y_tr, y_pseudo])
            sample_weights_tr = np.concatenate([sample_weights_tr, w_pseudo])
            
        # Evaluasi akhir pada validation fold yang steril
        final_clf = base_model_factory()
        if hasattr(final_clf, "fit") and "sample_weight" in final_clf.fit.__code__.co_varnames:
            final_clf.fit(current_X_tr, current_y_tr, sample_weight=sample_weights_tr)
        else:
            final_clf.fit(current_X_tr, current_y_tr)
            
        preds = final_clf.predict(X_va)
        f1 = f1_score(y_va, preds, average='macro')
        cv_scores.append(f1)
        
    return cv_scores, diagnostic_info
```

### 5.2 Grid Penyetelan Hiperparameter Semi-Supervised
Eksperimenkan kombinasi berikut untuk menemukan konfigurasi optimal:
* `confidence_threshold`: [0.80, 0.85, 0.90]
* `top_k_per_class`: [30, 60, 100]
* `pseudo_weight`: [0.3, 0.5, 0.8]

---

## Modul 6: Inferensi Akhir, Ekspor Prediksi, & Verifikasi

### 6.1 Latih Model Final pada Seluruh Data
```python
# Latih model ensemble terbaik dengan data latih penuh + pseudo-label terbaik
best_clf = get_ensemble()

# Terapkan pseudo-labeling sekali pada seluruh data berlabel
best_clf.fit(X_labeled_full, y_labeled_full)
probs_unlab = best_clf.predict_proba(X_unlabeled_full)
max_probs = np.max(probs_unlab, axis=1)
pred_labels = best_clf.classes_[np.argmax(probs_unlab, axis=1)]

# Seleksi kuota seimbang
best_indices = []
best_pseudo_labels = []
for c in best_clf.classes_:
    mask = (pred_labels == c) & (max_probs >= 0.85)
    idx = np.where(mask)[0]
    if len(idx) > 0:
        top_idx = idx[np.argsort(-max_probs[idx])][:60]
        best_indices.extend(top_idx)
        best_pseudo_labels.extend([c] * len(top_idx))

# Gabungkan data latih akhir
X_final_train = vstack([X_labeled_full, X_unlabeled_full[best_indices]])
y_final_train = np.concatenate([y_labeled_full, np.array(best_pseudo_labels)])
w_final = np.concatenate([np.ones(X_labeled_full.shape[0]), np.full(len(best_indices), 0.5)])

# Fit model produksi
final_production_model = get_ensemble()
final_production_model.fit(X_final_train, y_final_train, sample_weight=w_final)

# Prediksi data uji
test_predictions = final_production_model.predict(X_test_full)

# 6.2 Simpan submission.csv
submission_df = pd.DataFrame({
    'id': test_df['id'],
    'label': test_predictions
})
submission_df.to_csv('submission.csv', index=False)
print("File submission.csv berhasil disimpan! Distribusi prediksi test:")
print(submission_df['label'].value_counts())
```

### 6.3 Verifikasi Otomatis
Jalankan verifikasi sistem:
```bash
.venv/bin/python check_submission.py submission.csv
```
Pastikan konsol mencetak:
```
[OK] submission.csv valid (1246 baris)
```
