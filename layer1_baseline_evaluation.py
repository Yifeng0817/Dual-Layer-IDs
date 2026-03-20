"""
Layer 1 Baseline Evaluation — IoMT IDS
════════════════════════════════════════════════════════════════════════════════
Evaluates 3 individual baseline models using DEFAULT sklearn hyperparameters.
No hyperparameter tuning. No config.yaml. No ensemble.

Workflow — proper 80/20 stratified split on FULL dataset:
  Split  : 80% train / 20% test  (stratified, no overlap)
  Train  : benign rows from the 80% train split ONLY → model.fit()
  Test   : ALL rows from the 20% test split (benign + attack)
           → model.predict()  (completely unseen traffic)

This matches your partner's Layer 2 approach and ensures:
  - NO traffic appears in both train and test sets
  - Model is evaluated on genuinely unknown traffic
  - Results are directly comparable across all models

Models evaluated:
  1. Isolation Forest     — default (n_estimators=100, contamination='auto')
  2. Local Outlier Factor — default (n_neighbors=20,  contamination='auto')
  3. One-Class SVM        — default (nu=0.5, gamma='scale', kernel='rbf')

Usage:
  python layer1_baseline_evaluation.py \\
      --dataset data/processed/full_inference_cleaned_casing_preserved.csv

Author: Angela Yam Bao Hui
Student ID: 23WMR14647
"""

import os
import sys
import time
import threading
import warnings
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import psutil
import tracemalloc

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

warnings.filterwarnings("ignore")

sys.path.append("src")
from preprocessing.feature_extractor import FeatureExtractor

import logging
logging.disable(logging.CRITICAL)

# ════════════════════════════════════════════════════════════════════════════
# DEFAULT SKLEARN HYPERPARAMETERS  (NOT from config.yaml — pure defaults)
# ════════════════════════════════════════════════════════════════════════════

IF_DEFAULT_PARAMS = {
    "n_estimators": 100,
    "contamination": "auto",
    "max_samples":   "auto",
    "random_state":  42,
    "n_jobs":        -1,
}

LOF_DEFAULT_PARAMS = {
    "n_neighbors":   20,
    "contamination": "auto",
    "metric":        "minkowski",
    "novelty":       True,
    "n_jobs":        -1,
}

OCSVM_DEFAULT_PARAMS = {
    "kernel": "rbf",
    "nu":     0.5,
    "gamma":  "scale",
}

TRAIN_RATIO   = 0.8   # 80% of full dataset for training
RANDOM_STATE  = 42
CHUNK_SIZE    = 200_000


# ════════════════════════════════════════════════════════════════════════════
# CPU MONITOR
# ════════════════════════════════════════════════════════════════════════════

def _make_cpu_monitor():
    samples   = []
    stop_flag = {"stop": False}
    def _monitor():
        while not stop_flag["stop"]:
            samples.append(psutil.cpu_percent(interval=0.1))
    t = threading.Thread(target=_monitor, daemon=True)
    return samples, stop_flag, t


# ════════════════════════════════════════════════════════════════════════════
# DATA LOADING — 80/20 STRATIFIED SPLIT ON FULL DATASET
# ════════════════════════════════════════════════════════════════════════════

def load_and_split(dataset_path: str):
    """
    Proper 80/20 stratified split on the FULL dataset.

    Step 1: Read all row indices + labels (lightweight — no feature extraction)
    Step 2: Stratified 80/20 split → train_indices, test_indices
    Step 3: Extract features for train set (benign only) → fit scaler
    Step 4: Extract features for test set (all rows) → apply scaler

    Returns:
        X_train      — normalised benign rows from 80% split  → model.fit()
        X_test       — normalised ALL rows from 20% split     → model.predict()
        y_test       — binary labels for test set (0=benign, 1=attack)
        scaler       — fitted StandardScaler
        imputer      — fitted SimpleImputer
        feature_count
        split_info   — dict with counts for reporting
    """
    print("=" * 60)
    print("DATA LOADING — 80/20 STRATIFIED SPLIT")
    print("=" * 60)
    print(f"Dataset     : {dataset_path}")
    print(f"Train ratio : {TRAIN_RATIO:.0%}  |  Test ratio : {1-TRAIN_RATIO:.0%}")

    # ── Step 1: Read labels only to get indices ─────────────────────────────
    print("\nStep 1: Reading labels for split...")
    labels_series = pd.read_csv(dataset_path, usecols=["label"])["label"]
    total_rows    = len(labels_series)

    # Binary label — 0=benign, 1=attack
    y_all = np.where(
        labels_series.astype(str).str.strip().isin(["0", "0.0"]), 0, 1
    )
    del labels_series

    n_benign = int(np.sum(y_all == 0))
    n_attack = int(np.sum(y_all == 1))
    print(f"  Total rows  : {total_rows:,}")
    print(f"  Benign      : {n_benign:,}  ({n_benign/total_rows*100:.2f}%)")
    print(f"  Attack      : {n_attack:,}  ({n_attack/total_rows*100:.2f}%)")

    # ── Step 2: Split benign 80/20, ALL attacks go to test ─────────────────
    print("\nStep 2: Splitting benign 80/20, ALL attacks go to test...")
    np.random.seed(RANDOM_STATE)

    benign_idx = np.where(y_all == 0)[0]
    attack_idx = np.where(y_all == 1)[0]

    # Shuffle benign rows only, then split 80/20
    np.random.shuffle(benign_idx)

    n_benign_train   = int(len(benign_idx) * TRAIN_RATIO)
    train_benign_idx = benign_idx[:n_benign_train]   # 80% benign → model.fit()
    test_benign_idx  = benign_idx[n_benign_train:]   # 20% benign → test (unseen)

    # ALL attack rows go to test — none wasted in train split
    # Attack rows in a train split would be ignored anyway since
    # one-class models only train on benign data
    train_idx = np.sort(train_benign_idx)
    test_idx  = np.sort(np.concatenate([test_benign_idx, attack_idx]))

    y_test = y_all[test_idx]

    n_train_benign = len(train_benign_idx)
    n_test_benign  = len(test_benign_idx)
    n_test_attack  = len(attack_idx)

    print(f"  Train       : {n_train_benign:,} benign rows  (80%) → model.fit()")
    print(f"  Test benign : {n_test_benign:,} rows  (20% unseen benign)")
    print(f"  Test attack : {n_test_attack:,} rows  (ALL attacks — none wasted)")
    print(f"  Test total  : {len(test_idx):,} rows")
    print(f"  ✅ No overlap — model never sees any test row during training")

    # ── Step 3: Extract + normalise features for train benign rows ──────────
    print("\nStep 3: Extracting train features (benign only from 80%)...")

    extractor    = FeatureExtractor()
    meta_cols    = ["timestamp", "protocol", "label", "scenario"]
    train_benign_set = set(train_benign_idx.tolist())  # for fast lookup

    # Stream dataset, keep only train benign rows
    train_chunks = []
    all_raw_for_scaler = []  # collect all rows to fit scaler on full data
    row_offset   = 0

    reader = pd.read_csv(dataset_path, chunksize=CHUNK_SIZE)
    for chunk in reader:
        chunk_len = len(chunk)
        chunk_global_idx = np.arange(row_offset, row_offset + chunk_len)

        feat_chunk   = extractor.extract_from_dataframe(chunk)
        numeric_cols = [c for c in feat_chunk.columns if c not in meta_cols]
        X_raw = feat_chunk[numeric_cols].fillna(0.0).values.astype(np.float64)
        X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)

        # Collect all rows for scaler fitting
        all_raw_for_scaler.append(X_raw)

        # Keep only train benign rows for model training
        mask = np.array([i in train_benign_set for i in chunk_global_idx])
        if mask.sum() > 0:
            train_chunks.append(X_raw[mask])

        del chunk, feat_chunk
        row_offset += chunk_len

    # Fit imputer + scaler on full dataset (all rows)
    print("  Fitting imputer + scaler on full dataset...")
    X_full    = np.vstack(all_raw_for_scaler)
    del all_raw_for_scaler

    imputer = SimpleImputer(strategy="median")
    imputer.fit(X_full)
    X_full_imp = imputer.transform(X_full)
    del X_full

    scaler  = StandardScaler()
    scaler.fit(X_full_imp)
    del X_full_imp

    # Apply to train benign rows
    X_train_raw = np.vstack(train_chunks)
    del train_chunks
    X_train = scaler.transform(imputer.transform(X_train_raw))
    del X_train_raw

    print(f"  Training samples : {len(X_train):,}  (benign only, from 80% split)")

    # ── Step 4: Extract + normalise features for test rows ──────────────────
    print("\nStep 4: Extracting test features (20% split — all rows)...")

    test_idx_set = set(test_idx.tolist())
    test_chunks  = []
    row_offset   = 0

    reader = pd.read_csv(dataset_path, chunksize=CHUNK_SIZE)
    for chunk in reader:
        chunk_len        = len(chunk)
        chunk_global_idx = np.arange(row_offset, row_offset + chunk_len)

        feat_chunk   = extractor.extract_from_dataframe(chunk)
        numeric_cols = [c for c in feat_chunk.columns if c not in meta_cols]
        X_raw = feat_chunk[numeric_cols].fillna(0.0).values.astype(np.float64)
        X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)

        mask = np.array([i in test_idx_set for i in chunk_global_idx])
        if mask.sum() > 0:
            test_chunks.append(X_raw[mask])

        del chunk, feat_chunk
        row_offset += chunk_len

    X_test_raw = np.vstack(test_chunks)
    del test_chunks
    X_test = scaler.transform(imputer.transform(X_test_raw))
    del X_test_raw

    feat_count = X_train.shape[1]
    print(f"  Test samples     : {len(X_test):,}  "
          f"(benign={n_test_benign:,}, attack={n_test_attack:,})")
    print(f"  Feature count    : {feat_count}")
    print("=" * 60)

    split_info = {
        "total_rows":      total_rows,
        "n_train_benign":  n_train_benign,
        "n_train_attack":  0,               # no attacks in train (one-class)
        "n_test_benign":   n_test_benign,
        "n_test_attack":   n_test_attack,
    }

    return X_train, X_test, y_test, scaler, imputer, feat_count, split_info


# ════════════════════════════════════════════════════════════════════════════
# RESULT PRINTER
# ════════════════════════════════════════════════════════════════════════════

def print_results(
    model_name, y_true, y_pred,
    train_time_s, inference_time_s,
    peak_mem_mb, avg_cpu_pct, feature_count, split_info,
):
    total    = len(y_true)
    n_benign = int(np.sum(y_true == 0))
    n_attack = int(np.sum(y_true == 1))

    acc     = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    w_f1    = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    prec, rec, f1c, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], average=None, zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn = int(cm[0, 0]); fp = int(cm[0, 1])
    fn = int(cm[1, 0]); tp = int(cm[1, 1])

    latency_ms = (inference_time_s / total) * 1000 if total > 0 else 0
    throughput  = total / inference_time_s           if inference_time_s > 0 else 0

    SEP = "=" * 60
    print(f"\n{SEP}")
    print(f"  {model_name}")
    print(SEP)

    print(f"\n  Dataset Info  (80/20 stratified split)")
    print(f"  {'─'*40}")
    print(f"  Train (80%) benign used  : {split_info['n_train_benign']:,}")
    print(f"  Test  (20%) total        : {total:,}")
    print(f"    Benign (unseen)        : {n_benign:,}  ({n_benign/total*100:.1f}%)")
    print(f"    Attack (unseen)        : {n_attack:,}  ({n_attack/total*100:.1f}%)")
    print(f"  Feature count            : {feature_count}")

    print(f"\n  Resource & Performance")
    print(f"  {'─'*40}")
    print(f"  Training Time         : {train_time_s:.4f} s")
    print(f"  Total Inference Time  : {inference_time_s:.4f} s")
    print(f"  Avg Time per Sample   : {latency_ms/1000:.8f} s")
    print(f"  Latency per Sample    : {latency_ms:.4f} ms")
    print(f"  Throughput            : {throughput:,.0f} samples/sec")
    print(f"  Peak Memory Usage     : {peak_mem_mb:.2f} MB")
    print(f"  CPU Utilization (Avg) : {avg_cpu_pct:.2f} %")

    print(f"\n  Overall Performance")
    print(f"  {'─'*40}")
    print(f"  Accuracy              : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Balanced Accuracy     : {bal_acc:.4f}  ({bal_acc*100:.2f}%)")
    print(f"  Weighted F1-Score     : {w_f1:.4f}  ({w_f1*100:.2f}%)")

    print(f"\n  Confusion Matrix")
    print(f"  {'─'*40}")
    print(f"                         Predicted Benign   Predicted Attack")
    print(f"  Actual Benign          {tn:>14,}   {fp:>16,}   <- {n_benign:,} total")
    print(f"  Actual Attack          {fn:>14,}   {tp:>16,}   <- {n_attack:,} total")
    print(f"\n  TN (Benign correct)   : {tn:,}")
    print(f"  FP (Benign flagged)   : {fp:,}")
    print(f"  FN (Attack MISSED)    : {fn:,}  <- lower is better")
    print(f"  TP (Attack caught)    : {tp:,}")

    print(f"\n  Classification Report")
    print(f"  {'─'*40}")
    print(classification_report(
        y_true, y_pred,
        target_names=["Benign", "Attack"],
        digits=4, zero_division=0
    ))

    print(f"  Target Criteria")
    print(f"  {'─'*40}")
    for label, val, thr in [
        ("Attack Recall  >= 98%", rec[1], 0.98),
        ("Benign Recall  >= 90%", rec[0], 0.90),
        ("Weighted F1    >= 98%", w_f1,   0.98),
    ]:
        status = "PASS" if val >= thr else "FAIL"
        print(f"  {label} : {val*100:.2f}%  [{status}]")

    return {
        "model":             model_name,
        "feature_count":     feature_count,
        "n_train_benign":    split_info["n_train_benign"],
        "n_test_benign":     split_info["n_test_benign"],
        "n_test_attack":     split_info["n_test_attack"],
        "accuracy":          round(acc,          4),
        "balanced_accuracy": round(bal_acc,       4),
        "weighted_f1":       round(w_f1,          4),
        "benign_precision":  round(float(prec[0]),4),
        "benign_recall":     round(float(rec[0]), 4),
        "benign_f1":         round(float(f1c[0]), 4),
        "attack_precision":  round(float(prec[1]),4),
        "attack_recall":     round(float(rec[1]), 4),
        "attack_f1":         round(float(f1c[1]), 4),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "train_time_s":     round(train_time_s,     4),
        "inference_time_s": round(inference_time_s, 4),
        "latency_ms":       round(latency_ms,        6),
        "throughput":       round(throughput,         2),
        "peak_memory_mb":   round(peak_mem_mb,        2),
        "avg_cpu_pct":      round(avg_cpu_pct,        2),
    }


# ════════════════════════════════════════════════════════════════════════════
# MODEL RUNNER
# ════════════════════════════════════════════════════════════════════════════

def run_model(model_name, model, X_train, X_test, y_test,
              feature_count, split_info):
    print(f"\n{'='*60}")
    print(f"  Running: {model_name}")
    print(f"  Training on {len(X_train):,} benign samples (80% split)...")

    # Training
    tracemalloc.start()
    cpu_samples_tr, stop_flag_tr, cpu_thread_tr = _make_cpu_monitor()
    cpu_thread_tr.start()

    t0 = time.perf_counter()
    model.fit(X_train)
    train_time_s = time.perf_counter() - t0

    stop_flag_tr["stop"] = True
    cpu_thread_tr.join(timeout=2.0)
    _, train_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    train_cpu = float(np.mean(cpu_samples_tr)) if cpu_samples_tr else psutil.cpu_percent(interval=1)
    print(f"  Training complete : {train_time_s:.4f} s")

    # Inference
    cpu_samples_inf, stop_flag_inf, cpu_thread_inf = _make_cpu_monitor()
    cpu_thread_inf.start()
    tracemalloc.start()

    t0 = time.perf_counter()
    raw_preds = model.predict(X_test)
    inference_time_s = time.perf_counter() - t0

    _, inf_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    stop_flag_inf["stop"] = True
    cpu_thread_inf.join(timeout=2.0)

    y_pred = np.where(raw_preds == -1, 1, 0)
    peak_mem_mb = max(train_peak, inf_peak) / (1024 ** 2)
    avg_cpu_pct = max(
        float(np.mean(cpu_samples_tr))  if cpu_samples_tr  else 0,
        float(np.mean(cpu_samples_inf)) if cpu_samples_inf else 0
    )
    print(f"  Inference complete: {inference_time_s:.4f} s")

    return print_results(
        model_name, y_test, y_pred,
        train_time_s, inference_time_s,
        peak_mem_mb, avg_cpu_pct, feature_count, split_info,
    )


# ════════════════════════════════════════════════════════════════════════════
# COMPARISON TABLE
# ════════════════════════════════════════════════════════════════════════════

def print_comparison_table(results):
    W     = 90
    col_w = 24
    lbl_w = 24

    print("\n\n" + "=" * W)
    print("  LAYER 1 BASELINE COMPARISON SUMMARY")
    print("  Train: 80% benign | Test: 20% unseen benign + ALL attacks")
    print("=" * W)

    short = [r["model"].split("(")[0].strip() for r in results]
    print(f"  {'Metric':<{lbl_w}}" + "".join(f"{n:>{col_w}}" for n in short))
    print("  " + "-" * (W - 2))

    print(f"\n  -- Split Info")
    for label, key in [
        ("Train benign rows",  "n_train_benign"),
        ("Test benign rows",   "n_test_benign"),
        ("Test attack rows",   "n_test_attack"),
    ]:
        row = f"  {label:<{lbl_w}}" + "".join(f"{r[key]:>{col_w},}" for r in results)
        print(row)

    print(f"\n  -- Classification Metrics")
    for label, key in [
        ("Accuracy",          "accuracy"),
        ("Balanced Accuracy", "balanced_accuracy"),
        ("Weighted F1",       "weighted_f1"),
        ("Benign Precision",  "benign_precision"),
        ("Benign Recall",     "benign_recall"),
        ("Benign F1",         "benign_f1"),
        ("Attack Precision",  "attack_precision"),
        ("Attack Recall *",   "attack_recall"),
        ("Attack F1",         "attack_f1"),
    ]:
        row = f"  {label:<{lbl_w}}" + "".join(f"{r[key]:>{col_w}.4f}" for r in results)
        print(row)

    print(f"\n  -- Confusion Matrix")
    for label, key in [
        ("TN (Benign correct)", "tn"), ("FP (Benign flagged)", "fp"),
        ("FN (Attack MISSED)",  "fn"), ("TP (Attack caught)",  "tp"),
    ]:
        row = f"  {label:<{lbl_w}}" + "".join(f"{r[key]:>{col_w},}" for r in results)
        print(row)

    print(f"\n  -- Resource & Efficiency")
    for label, key in [
        ("Train Time (s)",      "train_time_s"),
        ("Inference Time (s)",  "inference_time_s"),
        ("Latency (ms/sample)", "latency_ms"),
        ("Throughput (samp/s)", "throughput"),
        ("Peak Memory (MB)",    "peak_memory_mb"),
        ("Avg CPU (%)",         "avg_cpu_pct"),
        ("Feature Count",       "feature_count"),
    ]:
        row = f"  {label:<{lbl_w}}"
        for r in results:
            val = r[key]
            if key == "throughput":
                row += f"{val:>{col_w},.2f}"
            elif key == "feature_count":
                row += f"{int(val):>{col_w}}"
            else:
                row += f"{val:>{col_w}.4f}"
        print(row)

    print(f"\n  -- Target Criteria")
    for label, key, thr in [
        ("Attack Recall >= 98%", "attack_recall", 0.98),
        ("Benign Recall >= 90%", "benign_recall", 0.90),
        ("Weighted F1   >= 98%", "weighted_f1",   0.98),
    ]:
        row = f"  {label:<{lbl_w}}" + "".join(
            f"{'PASS' if r[key] >= thr else 'FAIL':>{col_w}}" for r in results
        )
        print(row)

    print(f"\n  * Attack Recall is the primary evaluation criterion for an IDS.")
    print("=" * W + "\n")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Layer 1 Baseline Evaluation — 80/20 stratified split"
    )
    parser.add_argument("--dataset", required=True,
                        help="Path to the full inference CSV")
    parser.add_argument("--skip-ocsvm", action="store_true",
                        help="Skip OCSVM (slow on large datasets)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save JSON results")
    args = parser.parse_args()

    # Load and split data once — shared by all three models
    X_train, X_test, y_test, scaler, imputer, feat_count, split_info = \
        load_and_split(args.dataset)

    all_results = []

    # 1. Isolation Forest
    all_results.append(run_model(
        "Isolation Forest (Baseline - Default Params)",
        IsolationForest(**IF_DEFAULT_PARAMS),
        X_train, X_test, y_test, feat_count, split_info,
    ))

    # 2. Local Outlier Factor
    all_results.append(run_model(
        "Local Outlier Factor (Baseline - Default Params)",
        LocalOutlierFactor(**LOF_DEFAULT_PARAMS),
        X_train, X_test, y_test, feat_count, split_info,
    ))

    # 3. One-Class SVM
    if not args.skip_ocsvm:
        all_results.append(run_model(
            "One-Class SVM (Baseline - Default Params)",
            OneClassSVM(**OCSVM_DEFAULT_PARAMS),
            X_train, X_test, y_test, feat_count, split_info,
        ))
    else:
        print("\n[OCSVM] Skipped — run without --skip-ocsvm to include it")

    print_comparison_table(all_results)

    out = args.output or \
        f"results/layer1_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved -> {out}")


if __name__ == "__main__":
    main()