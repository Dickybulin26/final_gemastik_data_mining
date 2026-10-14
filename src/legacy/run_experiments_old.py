"""
run_experiments.py - Skrip Eksekusi Eksperimen Komparasi Baseline vs Semi-Supervised
"""
import re
import warnings
import numpy as np
import pandas as pd
from scipy.sparse import vstack
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

warnings.filterwarnings('ignore')

# 1. Pemuatan Data
print("Memuat dataset...")
train_df = pd.read_csv('data/labeled.csv')
unlabeled_df = pd.read_csv('data/unlabeled.csv')
test_df = pd.read_csv('data/test.csv')

print(f"Data berlabel   : {train_df.shape[0]} baris")
print(f"Data tanpa label: {unlabeled_df.shape[0]} baris")
print(f"Data uji        : {test_df.shape[0]} baris")

# 2. Text Preprocessing
SLANG_MAP = {
    'gak': 'tidak', 'ga': 'tidak', 'nggak': 'tidak', 'tdk': 'tidak',
    'bkn': 'bukan', 'bgt': 'banget', 'bgtt': 'banget',
    'tp': 'tetapi', 'dgn': 'dengan', 'sdh': 'sudah', 'blm': 'belum',
    'sy': 'saya', 'yg': 'yang', 'utk': 'untuk', 'aja': 'saja'
}

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'@\w+|#\w+', ' ', text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    words = text.split()
    words = [SLANG_MAP.get(w, w) for w in words]
    return ' '.join(words)

train_df['clean_text'] = train_df['text'].apply(clean_text)
unlabeled_df['clean_text'] = unlabeled_df['text'].apply(clean_text)
test_df['clean_text'] = test_df['text'].apply(clean_text)

# 3. Feature Extraction Setup
def build_vectorizer(min_df_w=2, min_df_c=3):
    word_vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=min_df_w,
        sublinear_tf=True
    )
    char_vec = TfidfVectorizer(
        ngram_range=(3, 5),
        analyzer='char',
        min_df=min_df_c,
        sublinear_tf=True
    )
    return FeatureUnion([
        ('word', word_vec),
        ('char', char_vec)
    ])

# Transductive Feature Fitting (Labeled + Unlabeled)
print("\nFitting Transductive Feature Extractor pada Labeled + Unlabeled (10.933 teks)...")
full_corpus = pd.concat([train_df['clean_text'], unlabeled_df['clean_text']], ignore_index=True)
transductive_vectorizer = build_vectorizer()
transductive_vectorizer.fit(full_corpus)

X_lab_trans = transductive_vectorizer.transform(train_df['clean_text'])
X_unlab_trans = transductive_vectorizer.transform(unlabeled_df['clean_text'])
X_test_trans = transductive_vectorizer.transform(test_df['clean_text'])
y_lab = train_df['label'].values

print(f"Dimensi Fitur Transductive: {X_lab_trans.shape[1]} fitur!")

# 4. Model Factories
def get_lr():
    return LogisticRegression(class_weight='balanced', C=1.0, max_iter=1000, random_state=42)

def get_svm():
    base = LinearSVC(class_weight='balanced', C=0.5, random_state=42)
    return CalibratedClassifierCV(base, cv=3)

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

SKF = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 5. Eksperimen 1: Pure Supervised (Inductive vs Transductive)
print("\n" + "="*70)
print("EKSPERIMEN BASELINE SUPERVISED (Stratified 5-Fold CV)")
print("="*70)

# A. Inductive (TF-IDF di-fit HANYA pada train fold berlabel)
def eval_inductive(model_factory):
    scores = []
    for tr_idx, va_idx in SKF.split(train_df, y_lab):
        vec = build_vectorizer(min_df_w=1, min_df_c=2)
        X_tr = vec.fit_transform(train_df['clean_text'].iloc[tr_idx])
        X_va = vec.transform(train_df['clean_text'].iloc[va_idx])
        clf = model_factory()
        clf.fit(X_tr, y_lab[tr_idx])
        preds = clf.predict(X_va)
        scores.append(f1_score(y_lab[va_idx], preds, average='macro'))
    return scores

# B. Transductive (TF-IDF di-fit pada Labeled + Unlabeled)
def eval_transductive(model_factory):
    scores = []
    for tr_idx, va_idx in SKF.split(X_lab_trans, y_lab):
        X_tr = X_lab_trans[tr_idx]
        X_va = X_lab_trans[va_idx]
        clf = model_factory()
        clf.fit(X_tr, y_lab[tr_idx])
        preds = clf.predict(X_va)
        scores.append(f1_score(y_lab[va_idx], preds, average='macro'))
    return scores

models = {
    "Logistic Regression (Balanced)": get_lr,
    "Calibrated LinearSVC (Balanced)": get_svm,
    "Complement Naive Bayes": get_cnb,
    "Soft Voting Ensemble": get_ensemble
}

print("\n--- 1. Pure Supervised Inductive (Hanya 300 label, tanpa corpus unlabeled) ---")
for name, factory in models.items():
    sc = eval_inductive(factory)
    print(f"{name:32s}: Mean={np.mean(sc):.4f} (±{np.std(sc):.4f}) | Folds: {[round(s, 4) for s in sc]}")

print("\n--- 2. Supervised Transductive (Kosakata TF-IDF diperluas dengan unlabeled) ---")
for name, factory in models.items():
    sc = eval_transductive(factory)
    print(f"{name:32s}: Mean={np.mean(sc):.4f} (±{np.std(sc):.4f}) | Folds: {[round(s, 4) for s in sc]}")

# 6. Eksperimen Semi-Supervised Pseudo-Labeling
print("\n" + "="*70)
print("EKSPERIMEN SEMI-SUPERVISED PSEUDO-LABELING")
print("="*70)

# A. Standard Naive Self-Training (Global Flat Threshold - Memperlihatkan Confirmation Bias)
def eval_naive_self_training(model_factory, threshold=0.85):
    scores = []
    pseudo_counts = []
    classes = np.unique(y_lab)
    for fold, (tr_idx, va_idx) in enumerate(SKF.split(X_lab_trans, y_lab)):
        X_tr = X_lab_trans[tr_idx]
        y_tr = y_lab[tr_idx]
        X_va = X_lab_trans[va_idx]
        y_va = y_lab[va_idx]
        
        clf = model_factory()
        clf.fit(X_tr, y_tr)
        probs = clf.predict_proba(X_unlab_trans)
        max_p = np.max(probs, axis=1)
        pred_c = classes[np.argmax(probs, axis=1)]
        
        mask = max_p >= threshold
        if fold == 0:
            counts = pd.Series(pred_c[mask]).value_counts().to_dict()
            pseudo_counts.append(counts)
            
        X_pseudo = X_unlab_trans[mask]
        y_pseudo = pred_c[mask]
        
        X_aug = vstack([X_tr, X_pseudo])
        y_aug = np.concatenate([y_tr, y_pseudo])
        
        clf_aug = model_factory()
        clf_aug.fit(X_aug, y_aug)
        preds = clf_aug.predict(X_va)
        scores.append(f1_score(y_va, preds, average='macro'))
    return scores, pseudo_counts[0] if pseudo_counts else {}

# B. Class-Balanced Adaptive Pseudo-Labeling (CB-PL)
def eval_cb_pseudo_labeling(model_factory, top_k=60, threshold=0.80, pseudo_weight=0.5):
    scores = []
    pseudo_counts = []
    classes = np.unique(y_lab)
    for fold, (tr_idx, va_idx) in enumerate(SKF.split(X_lab_trans, y_lab)):
        X_tr = X_lab_trans[tr_idx]
        y_tr = y_lab[tr_idx]
        X_va = X_lab_trans[va_idx]
        y_va = y_lab[va_idx]
        
        w_tr = np.ones(X_tr.shape[0])
        clf = model_factory()
        clf.fit(X_tr, y_tr, sample_weight=w_tr)
        
        probs = clf.predict_proba(X_unlab_trans)
        max_p = np.max(probs, axis=1)
        pred_c = classes[np.argmax(probs, axis=1)]
        
        chosen_indices = []
        chosen_labels = []
        chosen_weights = []
        
        for c in classes:
            c_mask = (pred_c == c) & (max_p >= threshold)
            c_idx = np.where(c_mask)[0]
            if len(c_idx) > 0:
                ranked = c_idx[np.argsort(-max_p[c_idx])][:top_k]
                chosen_indices.extend(ranked)
                chosen_labels.extend([c] * len(ranked))
                chosen_weights.extend([pseudo_weight] * len(ranked))
                
        if fold == 0:
            counts = pd.Series(chosen_labels).value_counts().to_dict()
            pseudo_counts.append(counts)
            
        if len(chosen_indices) > 0:
            X_pseudo = X_unlab_trans[chosen_indices]
            y_pseudo = np.array(chosen_labels)
            w_pseudo = np.array(chosen_weights)
            
            X_aug = vstack([X_tr, X_pseudo])
            y_aug = np.concatenate([y_tr, y_pseudo])
            w_aug = np.concatenate([w_tr, w_pseudo])
            
            clf_aug = model_factory()
            clf_aug.fit(X_aug, y_aug, sample_weight=w_aug)
            preds = clf_aug.predict(X_va)
        else:
            preds = clf.predict(X_va)
            
        scores.append(f1_score(y_va, preds, average='macro'))
    return scores, pseudo_counts[0] if pseudo_counts else {}

print("\n--- 3. Naive Flat-Threshold Self-Training (Threshold >= 0.85) ---")
naive_sc, naive_dist = eval_naive_self_training(get_ensemble, threshold=0.85)
print(f"Naive Self-Training Macro-F1: Mean={np.mean(naive_sc):.4f} (±{np.std(naive_sc):.4f}) | Folds: {[round(s, 4) for s in naive_sc]}")
print(f"Distribusi Pseudo-label (Fold 0): {naive_dist}")

print("\n--- 4. Class-Balanced Pseudo-Labeling (CB-PL) (Top-K per class, weight=0.5) ---")
for k in [30, 60, 100]:
    cb_sc, cb_dist = eval_cb_pseudo_labeling(get_ensemble, top_k=k, threshold=0.80, pseudo_weight=0.5)
    print(f"CB-PL (Top-{k} per class): Mean={np.mean(cb_sc):.4f} (±{np.std(cb_sc):.4f}) | Folds: {[round(s, 4) for s in cb_sc]}")
    if k == 60:
        print(f"Distribusi Pseudo-label (Fold 0, k=60): {cb_dist}")

print("\nEksperimen selesai!")
