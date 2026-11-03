"""
Offline training and comparative evaluation of NLP models on the authentic ISOT dataset.
- Stratified 80/20 train/test split (no data leakage).
- TF-IDF + Logistic Regression baseline trained on full ISOT train split.
- BERT contextual feature extractor + trained classification head on ISOT train split.
- Evaluates both models on held-out ISOT test split.
- Saves model artifacts and benchmark results for zero-latency dashboard and API usage.
"""
import os
import sys
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.logging_config import logger
from datasets.dataset_loader import DatasetLoader


def train_and_evaluate_isot(sample_bert_train: int = 3000, sample_bert_test: int = 1000):
    logger.info("Loading normalized ISOT dataset for training...")
    loader = DatasetLoader()
    df = loader.load_dataset("isot")
    logger.info(f"Loaded {len(df)} total ISOT records.")

    # Combine title and text for rich contextual representation
    texts = (df["title"].fillna("") + ". " + df["text"].fillna("")).tolist()
    labels = df["label"].astype(int).values

    # Stratified Train/Test split (80/20)
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        texts, labels, test_size=0.20, random_state=42, stratify=labels
    )
    logger.info(f"Split completed: Train={len(y_train)}, Test={len(y_test)}")

    # -------------------------------------------------------------
    # 1. TF-IDF + Logistic Regression Baseline
    # -------------------------------------------------------------
    logger.info("Training TF-IDF Baseline on full ISOT train set...")
    t0_tfidf = time.time()
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train_text)
    tfidf_clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    tfidf_clf.fit(X_train_vec, y_train)
    t1_tfidf = time.time()
    logger.info(f"TF-IDF training completed in {t1_tfidf - t0_tfidf:.2f}s.")

    # Evaluate TF-IDF on test set
    X_test_vec = vectorizer.transform(X_test_text)
    tfidf_probs = tfidf_clf.predict_proba(X_test_vec)[:, 1]
    tfidf_preds = (tfidf_probs >= 0.5).astype(int)

    tfidf_acc = float(accuracy_score(y_test, tfidf_preds))
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, tfidf_preds, average="binary")
    tfidf_auc = float(roc_auc_score(y_test, tfidf_probs))
    tfidf_cm = confusion_matrix(y_test, tfidf_preds).tolist()

    tfidf_metrics = {
        "model": "TF-IDF + Logistic Regression",
        "accuracy": round(tfidf_acc, 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(tfidf_auc, 4),
        "confusion_matrix": tfidf_cm,
        "train_samples": len(y_train),
        "test_samples": len(y_test),
        "training_time_sec": round(t1_tfidf - t0_tfidf, 2),
    }
    logger.info(f"TF-IDF Baseline Results: Accuracy={tfidf_acc:.4f}, F1={f1:.4f}, AUC={tfidf_auc:.4f}")

    # Save TF-IDF Model
    tfidf_path = PROJECT_ROOT / "models" / "nlp" / "tfidf_model.pkl"
    joblib.dump({"vectorizer": vectorizer, "model": tfidf_clf}, tfidf_path)
    logger.info(f"Saved trained TF-IDF model -> {tfidf_path}")

    # -------------------------------------------------------------
    # 2. BERT Contextual Language Model & Linear Probe
    # -------------------------------------------------------------
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    model_name = "bert-base-uncased"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading BERT backbone '{model_name}' on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bert_encoder = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, output_hidden_states=True
    ).to(device)
    bert_encoder.eval()

    # Stratified subsample for BERT contextual embedding extraction on CPU
    rng = np.random.default_rng(42)
    bert_train_idx = []
    for cls in [0, 1]:
        cls_idx = np.where(y_train == cls)[0]
        chosen = rng.choice(cls_idx, size=min(sample_bert_train // 2, len(cls_idx)), replace=False)
        bert_train_idx.extend(chosen)
    rng.shuffle(bert_train_idx)

    bert_test_idx = []
    for cls in [0, 1]:
        cls_idx = np.where(y_test == cls)[0]
        chosen = rng.choice(cls_idx, size=min(sample_bert_test // 2, len(cls_idx)), replace=False)
        bert_test_idx.extend(chosen)
    rng.shuffle(bert_test_idx)

    train_sub_texts = [X_train_text[i] for i in bert_train_idx]
    train_sub_labels = y_train[bert_train_idx]
    test_sub_texts = [X_test_text[i] for i in bert_test_idx]
    test_sub_labels = y_test[bert_test_idx]

    logger.info(f"Extracting BERT [CLS] embeddings for {len(train_sub_texts)} train articles...")
    t0_bert = time.time()

    def get_embeddings(texts_batch, batch_size=32):
        all_embeddings = []
        for i in range(0, len(texts_batch), batch_size):
            chunk = texts_batch[i:i + batch_size]
            inputs = tokenizer(
                chunk, padding=True, truncation=True, max_length=128, return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                out = bert_encoder(**inputs)
                cls_emb = out.hidden_states[-1][:, 0, :].cpu().numpy()
            all_embeddings.append(cls_emb)
        return np.vstack(all_embeddings)

    X_train_bert = get_embeddings(train_sub_texts, batch_size=32)
    X_test_bert = get_embeddings(test_sub_texts, batch_size=32)

    logger.info(f"Fitting Logistic Regression probe on {X_train_bert.shape} BERT representations...")
    bert_probe = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    bert_probe.fit(X_train_bert, train_sub_labels)
    t1_bert = time.time()

    bert_probs = bert_probe.predict_proba(X_test_bert)[:, 1]
    bert_preds = (bert_probs >= 0.5).astype(int)

    bert_acc = float(accuracy_score(test_sub_labels, bert_preds))
    b_prec, b_rec, b_f1, _ = precision_recall_fscore_support(test_sub_labels, bert_preds, average="binary")
    bert_auc = float(roc_auc_score(test_sub_labels, bert_probs))
    bert_cm = confusion_matrix(test_sub_labels, bert_preds).tolist()

    bert_metrics = {
        "model": "BERT (bert-base-uncased) + ISOT-Trained Linear Probe",
        "accuracy": round(bert_acc, 4),
        "precision": round(float(b_prec), 4),
        "recall": round(float(b_rec), 4),
        "f1_score": round(float(b_f1), 4),
        "roc_auc": round(bert_auc, 4),
        "confusion_matrix": bert_cm,
        "train_samples": len(train_sub_labels),
        "test_samples": len(test_sub_labels),
        "training_time_sec": round(t1_bert - t0_bert, 2),
    }
    logger.info(f"BERT Results: Accuracy={bert_acc:.4f}, F1={b_f1:.4f}, AUC={bert_auc:.4f}")

    # Save BERT Probe
    probe_path = PROJECT_ROOT / "models" / "nlp" / "bert_probe.pkl"
    joblib.dump(bert_probe, probe_path)
    logger.info(f"Saved trained BERT probe -> {probe_path}")

    # Save Comparative Benchmark JSON
    benchmarks = {
        "dataset": "ISOT Fake and Real News",
        "total_records": len(df),
        "train_set_size": len(y_train),
        "test_set_size": len(y_test),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": {
            "tfidf_logistic_regression": tfidf_metrics,
            "bert_transformer": bert_metrics,
        }
    }
    benchmarks_path = PROJECT_ROOT / "reports" / "model_benchmarks.json"
    benchmarks_path.parent.mkdir(parents=True, exist_ok=True)
    with open(benchmarks_path, "w", encoding="utf-8") as f:
        json.dump(benchmarks, f, indent=2)
    logger.info(f"Saved benchmark results -> {benchmarks_path}")

    return benchmarks


if __name__ == "__main__":
    train_and_evaluate_isot(sample_bert_train=2000, sample_bert_test=500)
