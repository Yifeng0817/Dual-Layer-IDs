"""
IoMT Intrusion Detection System with Baseline Profiling
Scenario-based baseline profiling (NOT device-based)

Author: Angela Yam Bao Hui
Student ID: 23WMR14647
"""

import pandas as pd
import numpy as np
import psutil
import yaml
import joblib
import json
import logging
import argparse
import copy
from pathlib import Path
from datetime import datetime
import sys
import time
from collections import defaultdict

from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, balanced_accuracy_score,
    precision_recall_fscore_support, f1_score
)

sys.path.append('src')

from preprocessing.feature_extractor import FeatureExtractor
from detection.ensemble_detector import EnsembleDetector
from detection.baseline_profiler import BaselineProfiler
from database.db_connector import DatabaseConnector


Path('logs').mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/main.log', encoding='utf-8'),
    ]
)
logger = logging.getLogger(__name__)


def cprint(msg: str, level: str = 'info'):
    print(msg)
    if level == 'info':    logger.info(msg)
    elif level == 'error': logger.error(msg)
    elif level == 'warn':  logger.warning(msg)




class PerformanceTimer:
    """Track execution time for different stages"""
    def __init__(self):
        self.timings = defaultdict(list)
        self.start_time = None
        self.stage_name = None
    
    def start(self, stage_name):
        """Start timing a stage"""
        self.stage_name = stage_name
        self.start_time = time.time()
    
    def stop(self):
        """Stop timing and record"""
        if self.start_time and self.stage_name:
            elapsed = time.time() - self.start_time
            self.timings[self.stage_name].append(elapsed)
            self.start_time = None
            return elapsed
        return 0
    
    def get_summary(self):
        """Get timing summary"""
        summary = {}
        for stage, times in self.timings.items():
            summary[stage] = {
                'total': sum(times),
                'avg': np.mean(times),
                'min': np.min(times),
                'max': np.max(times),
                'count': len(times)
            }
        return summary
    
    def print_summary(self, total_samples):
        """Print formatted timing summary"""
        logger.info("\n" + "="*80)
        logger.info("⚡ PERFORMANCE ANALYSIS")
        logger.info("="*80)
        
        summary = self.get_summary()
        total_time = sum(s['total'] for s in summary.values())
        
        logger.info(f"\n📊 Execution Time Breakdown:")
        logger.info("─" * 80)
        
        for stage, stats in sorted(summary.items(), key=lambda x: x[1]['total'], reverse=True):
            percentage = (stats['total'] / total_time * 100) if total_time > 0 else 0
            logger.info(f"\n  {stage}:")
            logger.info(f"    Total:   {stats['total']:.3f}s ({percentage:.1f}%)")
            logger.info(f"    Average: {stats['avg']:.3f}s")
            if stats['count'] > 1:
                logger.info(f"    Min/Max: {stats['min']:.3f}s / {stats['max']:.3f}s")
        
        logger.info(f"\n📈 Overall Performance:")
        logger.info("─" * 80)
        logger.info(f"  Total Time:        {total_time:.2f} seconds")
        logger.info(f"  Total Samples:     {total_samples:,}")
        
        if total_time > 0:
            logger.info(f"  Throughput:        {total_samples/total_time:.0f} samples/second")
            logger.info(f"  Avg Latency:       {(total_time/total_samples)*1000:.2f} ms/sample")
        
       
        logger.info(f"\n🎯 Real-Time Capability Assessment:")
        logger.info("─" * 80)
        
        if total_time > 0:
            throughput = total_samples / total_time
            latency_ms = (total_time / total_samples) * 1000
            
            if throughput >= 10000:
                logger.info(f"  ✅ EXCELLENT: {throughput:.0f} samples/sec (>10,000 target)")
            elif throughput >= 1000:
                logger.info(f"  ✅ GOOD: {throughput:.0f} samples/sec (>1,000 minimum)")
            else:
                logger.info(f"  ⚠️  SLOW: {throughput:.0f} samples/sec (<1,000 minimum)")
            
            if latency_ms < 10:
                logger.info(f"  ✅ EXCELLENT: {latency_ms:.2f} ms/sample (<10ms target)")
            elif latency_ms < 100:
                logger.info(f"  ✅ GOOD: {latency_ms:.2f} ms/sample (<100ms minimum)")
            else:
                logger.info(f"  ⚠️  HIGH LATENCY: {latency_ms:.2f} ms/sample (>100ms)")
        
        logger.info("="*80)


def load_config(config_path: str = 'config/config.yaml') -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def clear_baseline_cache(db: DatabaseConnector):
    """
    Clear all baseline-related cache entries from Redis
    """
    if db.redis_client:
        try:
            patterns = [
                'iomt:baseline:*',
                'iomt:baseline:scenario:*',
                'iomt:baseline:stats:*'
            ]
            
            total_deleted = 0
            for pattern in patterns:
                keys = db.redis_client.keys(pattern)
                if keys:
                    db.redis_client.delete(*keys)
                    total_deleted += len(keys)
            
            if total_deleted > 0:
                logger.info(f"  ✅ Cleared {total_deleted} baseline cache entries from Redis")
            else:
                logger.info(f"  ✅ Redis cache is clean (no baseline entries found)")
                
        except Exception as e:
            logger.warning(f"  ⚠️  Could not clear Redis cache: {e}")
    else:
        logger.warning(f"  ⚠️  Redis client not available - cache not cleared")


def reset_database(db: DatabaseConnector):
    """
    ✅ NEW: Reset database tables (clear all data and reset sequences)
    """
    logger.info("\n[DATABASE RESET]")
    logger.info("─" * 40)
    
    conn = None
    try:
        conn = db._get_connection()
        cursor = conn.cursor()
        
       
        logger.info("  Deleting existing data...")
        cursor.execute("DELETE FROM alerts")
        cursor.execute("DELETE FROM featured_traffic")
        cursor.execute("DELETE FROM scenario_baselines")
        cursor.execute("DELETE FROM baselines")
        
    
        logger.info("  Resetting sequences...")
        cursor.execute("SELECT setval('baselines_baseline_id_seq', 1, false)")
        cursor.execute("SELECT setval('scenario_baselines_baseline_stat_id_seq', 1, false)")
        cursor.execute("SELECT setval('featured_traffic_traffic_id_seq', 1, false)")
        cursor.execute("SELECT setval('alerts_alert_id_seq', 1, false)")
        
        conn.commit()
        cursor.close()
        
        logger.info("  ✅ Database reset complete!")
        logger.info("─" * 40)
        
        return True
        
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"  ❌ Database reset failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if conn:
            db._return_connection(conn)


def verify_database_state(db: DatabaseConnector):
    """
    Verify database tables are accessible and show current state
    """
    logger.info("\n[DATABASE STATE CHECK]")
    logger.info("─" * 40)
    
    conn = None
    try:
        conn = db._get_connection()
        cursor = conn.cursor()
        
      
        cursor.execute("SELECT current_database(), current_user")
        db_info = cursor.fetchone()
        logger.info(f"  Connected to database:    {db_info[0]}")
        logger.info(f"  Connected as user:        {db_info[1]}")
        
       
        cursor.execute("SELECT COUNT(*) FROM baselines")
        result = cursor.fetchone()
        baseline_count = result[0] if result else 0
        logger.info(f"  Baselines table:          {baseline_count} rows")
        
      
        cursor.execute("SELECT COUNT(*) FROM scenario_baselines")
        result = cursor.fetchone()
        stats_count = result[0] if result else 0
        logger.info(f"  Scenario_baselines table: {stats_count} rows")
        
   
        if baseline_count > 0:
            cursor.execute("SELECT baseline_id, scenario_name FROM baselines LIMIT 5")
            existing = cursor.fetchall()
            logger.info(f"  Existing baselines:")
            for bid, sname in existing:
                logger.info(f"    - ID {bid}: {sname[:40]}")
        
      
        try:
            cursor.execute("SELECT last_value FROM baselines_baseline_id_seq")
            seq_info = cursor.fetchone()
            
            if seq_info:
                last_val = seq_info[0] if seq_info[0] is not None else 1
                logger.info(f"  Baseline sequence:        last_value={last_val}")
            else:
                logger.info(f"  Baseline sequence:        (ready, next ID will be 1)")
        except Exception as e:
            logger.info(f"  Baseline sequence:        (ready for use)")
        
        cursor.close()
        
        logger.info("─" * 40)
        
       
        return baseline_count, stats_count
        
    except Exception as e:
        logger.error(f"  ❌ Database check failed: {e}")
        import traceback
        traceback.print_exc()
        return -1, -1
    finally:
        if conn:
            db._return_connection(conn)



def train_mode_benign_only(config: dict, dataset_path: str, sample_size: int = None, auto_reset: bool = True):
    """
    Training Mode — 80/20 benign split.
    Train on 80% of benign rows, hold out 20% for detection evaluation.
    All attack rows go to the test set (none wasted in train split).
    Console output is clean (no timestamps) — logs written to logs/main.log.
    """
    import time as _time
    t_total_start = _time.perf_counter()

    print("\n" + "="*60)
    print("TRAINING MODE: BASELINE PROFILING (Scenario-Based)")
    print("="*60)


    db = DatabaseConnector(config)
    baseline_count, stats_count = verify_database_state(db)

    if baseline_count > 0 or stats_count > 0:
        if auto_reset:
            print("\n⚠️  Existing data found — resetting database...")
            reset_database(db)
            clear_baseline_cache(db)
            baseline_count, stats_count = verify_database_state(db)
            if baseline_count > 0 or stats_count > 0:
                print("❌ Database reset failed! Please reset manually.")
                return
        else:
            print("\n⚠️  Database has existing data. Use --reset flag to clear.")
            return
    else:
        print("\n✅ Database is clean and ready for training")
        clear_baseline_cache(db)

    extractor    = FeatureExtractor()
    profiler     = BaselineProfiler(db, config)
    train_config = copy.deepcopy(config)

    LOAD_CHUNK_SIZE = 200_000
    RANDOM_STATE    = 42
    TRAIN_RATIO     = 0.8

    
    print("\n[STEP 1] Building 80/20 benign split...")
    t0 = _time.perf_counter()

    labels_series = pd.read_csv(dataset_path, usecols=['label'])['label']
    total_rows    = len(labels_series)
    y_all = np.where(
        labels_series.astype(str).str.strip().isin(['0', '0.0']), 0, 1
    )
    del labels_series

    benign_idx = np.where(y_all == 0)[0]
    attack_idx = np.where(y_all == 1)[0]
    np.random.seed(RANDOM_STATE)
    np.random.shuffle(benign_idx)

    n_benign_train   = int(len(benign_idx) * TRAIN_RATIO)
    train_benign_idx = set(benign_idx[:n_benign_train].tolist())
    test_benign_idx  = benign_idx[n_benign_train:]

    n_train_benign = len(train_benign_idx)
    n_test_benign  = len(test_benign_idx)
    n_test_attack  = len(attack_idx)

    print(f"  Total rows         : {total_rows:,}")
    print(f"  Benign total       : {len(benign_idx):,}")
    print(f"  Attack total       : {len(attack_idx):,}")
    print(f"  Train (80% benign) : {n_train_benign:,} rows  → model.fit()")
    print(f"  Test  (20% benign) : {n_test_benign:,} rows   (unseen benign)")
    print(f"  Test  (all attacks): {n_test_attack:,} rows   (all attacks)")
    print(f"  Elapsed            : {_time.perf_counter()-t0:.2f}s")

   
    print("\n[STEP 2] Feature Extraction + Scaler Fitting (chunked)...")
    t0 = _time.perf_counter()

    from sklearn.preprocessing import StandardScaler as _SS
    scaler      = _SS()
    all_raw     = []
    row_offset  = 0
    reader      = pd.read_csv(dataset_path, chunksize=LOAD_CHUNK_SIZE)

    for chunk in reader:
        feat_chunk = extractor.extract_from_dataframe(chunk)
        meta_cols  = ['timestamp', 'protocol', 'label', 'scenario']
        num_cols   = [c for c in feat_chunk.columns if c not in meta_cols]
        X_raw      = feat_chunk[num_cols].fillna(0.0).values.astype(np.float64)
        X_raw      = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)
        chunk_idx  = np.arange(row_offset, row_offset + len(chunk))
        all_raw.append((X_raw, chunk['label'].values, chunk_idx,
                        feat_chunk[meta_cols + num_cols] if 'scenario' in feat_chunk.columns else feat_chunk))
        row_offset += len(chunk)
        del chunk, feat_chunk

    X_full = np.vstack([c[0] for c in all_raw])
    scaler.fit(X_full)
    del X_full
    print(f"  Scaler fitted on {total_rows:,} rows  ({_time.perf_counter()-t0:.2f}s)")

  
    print("\n[STEP 3] Building train/test sets...")
    t0 = _time.perf_counter()

    benign_feat_chunks = []
    meta_cols = ['timestamp', 'protocol', 'label', 'scenario']

    for X_raw, labels, chunk_idx, feat_chunk in all_raw:
        X_scaled   = scaler.transform(X_raw)
        num_cols   = [c for c in feat_chunk.columns if c not in meta_cols]
        feat_chunk = feat_chunk.copy()
        feat_chunk[num_cols] = X_scaled
        feat_chunk['label']  = feat_chunk['label'].astype(str).str.strip().str.lower()

       
        train_mask = np.array([i in train_benign_idx for i in chunk_idx])
        if train_mask.sum() > 0:
            benign_feat_chunks.append(feat_chunk[train_mask])
        del feat_chunk, X_raw, X_scaled

    del all_raw
    benign_df  = pd.concat(benign_feat_chunks, ignore_index=True)
    del benign_feat_chunks
    attack_df  = pd.DataFrame()  # not needed for one-class training

    print(f"  Train benign df    : {len(benign_df):,} rows")
    print(f"  Elapsed            : {_time.perf_counter()-t0:.2f}s")

   
    print("\n[STEP 4] Scenario Identification...")
    t0 = _time.perf_counter()

    if 'scenario' not in benign_df.columns:
        print("❌ No scenario column in dataset!")
        print("   Run: python add_scenario_column.py --input", dataset_path, "--output", dataset_path)
        return

    baseline_mapping = profiler.identify_scenarios(benign_df)
    print(f"  ✅ Identified {len(baseline_mapping)} scenarios  ({_time.perf_counter()-t0:.2f}s)")

   
    conn = db._get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT baseline_id, scenario_name FROM baselines ORDER BY baseline_id")
        db_baselines = cursor.fetchall()
        cursor.close()
        for scenario_name, expected_id in baseline_mapping.items():
            if not any(bid == expected_id for bid, _ in db_baselines):
                raise Exception(f"Baseline {expected_id} not found in database!")
        print(f"  ✅ All {len(baseline_mapping)} baselines verified in database")
    except Exception as e:
        print(f"❌ Baseline verification failed: {e}")
        raise
    finally:
        db._return_connection(conn)

 
    print("\n[STEP 5] Building Scenario Baselines...")
    t0 = _time.perf_counter()
    feature_names = [c for c in benign_df.columns
                     if c not in ['timestamp', 'protocol', 'label', 'scenario']]
    profiler.build_all_baselines(benign_df, baseline_mapping, feature_names)
    print(f"  ✅ Baselines built  ({_time.perf_counter()-t0:.2f}s)")

  
    print("\n[STEP 6] Training Ensemble (80% benign)...")
    t0 = _time.perf_counter()

    semi_cfg = config.get('training', {}).get('semi_supervised', {})
    use_semi = semi_cfg.get('enabled', False)

    if use_semi:
        print("  Semi-supervised mode not supported with 80/20 split — using benign-only")

    ensemble = EnsembleDetector(train_config)
    X_train  = extractor.get_feature_vector(benign_df)
    ensemble.fit(X_train)
    train_time_s = _time.perf_counter() - t0

    print(f"  ✅ Ensemble trained on {len(X_train):,} samples  ({train_time_s:.2f}s)")

  
    print("\n[STEP 7] Saving Models...")
    models_dir = Path('data/models')
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(ensemble, models_dir / 'ensemble.joblib')
    joblib.dump(scaler,   models_dir / 'scaler.joblib')
    with open(models_dir / 'baseline_mapping.json', 'w', encoding='utf-8') as f:
        json.dump(baseline_mapping, f, indent=2)
  
    split_info = {
        'n_train_benign': n_train_benign,
        'n_test_benign':  n_test_benign,
        'n_test_attack':  n_test_attack,
        'train_ratio':    TRAIN_RATIO,
        'random_state':   RANDOM_STATE,
    }
    with open(models_dir / 'split_info.json', 'w', encoding='utf-8') as f:
        json.dump(split_info, f, indent=2)
    print(f"  ✅ Models saved to {models_dir}/")

    total_elapsed = _time.perf_counter() - t_total_start
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"  Training samples  : {len(X_train):,}  (80% benign)")
    print(f"  Features          : {len(feature_names)}")
    print(f"  Scenarios         : {len(baseline_mapping)}")
    print(f"  Total time        : {total_elapsed:.2f}s")
    print(f"  Log file          : logs/main.log")
    print("="*60)

    verify_database_state(db)
    db.close()


def detect_mode_with_metrics(config: dict, dataset_path: str, sample_size: int = None):
    """
    Detection Mode — evaluates on 20% unseen benign + ALL attacks.
    Uses the split indices saved during training.
    Clean console output (no timestamps) — logs written to logs/main.log.
    Output format matches layer1_ensemble_evaluation.py.
    """
    import time as _time
    import tracemalloc as _tracemalloc
    import threading as _threading
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score,
        classification_report, confusion_matrix,
        f1_score, precision_recall_fscore_support
    )

   
    def _make_cpu_monitor():
        samples   = []
        stop_flag = {"stop": False}
        def _monitor():
            while not stop_flag["stop"]:
                samples.append(psutil.cpu_percent(interval=0.1))
        t = _threading.Thread(target=_monitor, daemon=True)
        return samples, stop_flag, t

    t_total_start = _time.perf_counter()

    print("\n" + "="*60)
    print("DETECTION MODE")
    print("="*60)

    
    print("\n[STEP 1] Loading Models...")
    models_dir = Path('data/models')
    try:
        ensemble         = joblib.load(models_dir / 'ensemble.joblib')
        scaler           = joblib.load(models_dir / 'scaler.joblib')
        with open(models_dir / 'baseline_mapping.json') as f:
            baseline_mapping = json.load(f)
        with open(models_dir / 'split_info.json') as f:
            split_info = json.load(f)
        print(f"  ✅ Models loaded from {models_dir}/")
        print(f"  Train benign used : {split_info['n_train_benign']:,}  (80%)")
        print(f"  Expected test     : {split_info['n_test_benign']:,} unseen benign "
              f"+ {split_info['n_test_attack']:,} attacks")
    except FileNotFoundError as e:
        print(f"  ❌ Model not found: {e}")
        print("  Run: python main.py --mode train first")
        return

   
    print("\n[STEP 2] Rebuilding split indices...")
    t0 = _time.perf_counter()

    TRAIN_RATIO  = split_info['train_ratio']
    RANDOM_STATE = split_info['random_state']
    CHUNK_SIZE   = 500_000

    labels_series = pd.read_csv(dataset_path, usecols=['label'])['label']
    total_rows    = len(labels_series)
    if sample_size:
        total_rows = min(total_rows, sample_size)

    y_all = np.where(
        labels_series.astype(str).str.strip().isin(['0', '0.0']), 0, 1
    )
    del labels_series

    benign_idx = np.where(y_all == 0)[0]
    attack_idx = np.where(y_all == 1)[0]
    np.random.seed(RANDOM_STATE)
    np.random.shuffle(benign_idx)

    n_benign_train   = int(len(benign_idx) * TRAIN_RATIO)
    test_benign_idx  = set(benign_idx[n_benign_train:].tolist())
    test_attack_set  = set(attack_idx.tolist())
    test_idx_set     = test_benign_idx | test_attack_set

    print(f"  Test benign : {len(test_benign_idx):,} rows")
    print(f"  Test attack : {len(test_attack_set):,} rows")
    print(f"  Total test  : {len(test_idx_set):,} rows")
    print(f"  Elapsed     : {_time.perf_counter()-t0:.2f}s")

    
    print("\n[STEP 3] Running Detection (chunked)...")
    print(f"  Chunk size  : {CHUNK_SIZE:,} rows")

    extractor  = FeatureExtractor()
    meta_cols  = ['timestamp', 'protocol', 'label', 'scenario']

    all_y_true  = []; all_scores = []
    all_ocsvm   = []; all_lof    = []
    all_if_pred = []; all_voting = []

  
    _tracemalloc.start()
    cpu_samples, cpu_stop, cpu_thread = _make_cpu_monitor()
    cpu_thread.start()

    spec = config.get('ensemble', {}).get('specialization', {})
    if spec.get('enabled', False):
        eff_thr = 0.25 if spec.get('suspicious_as', 'attack') == 'attack' else 0.75
    else:
        eff_thr = config.get('ensemble', {}).get('threshold_low', 0.1)

    inf_time   = 0.0
    chunk_num  = 0
    rows_done  = 0
    row_offset = 0

    reader = pd.read_csv(dataset_path, chunksize=CHUNK_SIZE)
    for chunk_df in reader:
        if sample_size and row_offset >= sample_size:
            break

        chunk_len        = len(chunk_df)
        chunk_global_idx = np.arange(row_offset, row_offset + chunk_len)

        # Keep only test set rows
        test_mask = np.array([i in test_idx_set for i in chunk_global_idx])
        if test_mask.sum() == 0:
            row_offset += chunk_len
            del chunk_df
            continue

        chunk_num += 1
        chunk_df_test = chunk_df[test_mask].reset_index(drop=True)

        
        feat_chunk = extractor.extract_from_dataframe(chunk_df_test)
        num_cols   = [c for c in feat_chunk.columns if c not in meta_cols]
        raw_labels = feat_chunk['label'].astype(str).str.strip()
        y_chunk    = np.where(raw_labels.isin(['0', '0.0']), 0, 1)
        all_y_true.append(y_chunk)

       
        feat_norm, _ = extractor.normalize_features(feat_chunk, scaler)
        X_chunk      = extractor.get_feature_vector(feat_norm)

       
        t0         = _time.perf_counter()
        ocsvm_raw  = -ensemble.ocsvm.decision_function(X_chunk)
        lof_raw    = -ensemble.lof.decision_function(X_chunk)
        if_raw     = -ensemble.isolation_forest.decision_function(X_chunk)
        inf_time  += _time.perf_counter() - t0

       
        def _norm(s):
            p1, p99 = np.percentile(s, 1), np.percentile(s, 99)
            c = np.clip(s, p1, p99)
            return (c - p1) / (p99 - p1) if p99 - p1 > 1e-6 else np.zeros_like(s)

        ocsvm_norm = _norm(ocsvm_raw)
        lof_norm   = _norm(lof_raw)
        if_norm    = _norm(if_raw)

        ocsvm_pred = np.where(ocsvm_raw > 0, -1, 1)
        lof_pred   = np.where(lof_raw   > 0, -1, 1)
        if_pred    = np.where(if_raw    > 0, -1, 1)
        voting     = ((ocsvm_pred == -1).astype(int) +
                      (lof_pred   == -1).astype(int) +
                      (if_pred    == -1).astype(int))

        
        if spec.get('enabled', False):
            gk  = ensemble.gatekeeper_name if hasattr(ensemble, 'gatekeeper_name') else 'ocsvm'
            hn  = ensemble.hunter_name     if hasattr(ensemble, 'hunter_name')     else 'isolation_forest'
            gk_pred = ocsvm_pred if gk == 'ocsvm' else (lof_pred if gk == 'lof' else if_pred)
            hn_pred = if_pred    if hn == 'isolation_forest' else (lof_pred if hn == 'lof' else ocsvm_pred)
            scores = np.where((gk_pred == -1) & (hn_pred == -1), 1.0,
                     np.where((gk_pred ==  1) & (hn_pred == -1), 0.5, 0.0))
        else:
            w = config.get('ensemble', {}).get('weights', {})
            scores = (w.get('ocsvm', 0.333) * ocsvm_norm +
                      w.get('lof',   0.333) * lof_norm   +
                      w.get('isolation_forest', 0.333) * if_norm)

        all_scores.append(scores)
        all_ocsvm.append(ocsvm_pred)
        all_lof.append(lof_pred)
        all_if_pred.append(if_pred)
        all_voting.append(voting)

        rows_done += test_mask.sum()
        del chunk_df, chunk_df_test, feat_chunk, feat_norm, X_chunk
        row_offset += chunk_len

        print(f"  Chunk {chunk_num}: {rows_done:,} test rows processed")

  
    cpu_stop["stop"] = True
    cpu_thread.join(timeout=2.0)
    _, peak_mem = _tracemalloc.get_traced_memory()
    _tracemalloc.stop()

    peak_mem_mb = peak_mem / (1024 ** 2)
    avg_cpu_pct = float(np.mean(cpu_samples)) if cpu_samples else psutil.cpu_percent(interval=1)

  
    y_true          = np.concatenate(all_y_true)
    ensemble_scores = np.concatenate(all_scores)
    ocsvm_preds     = np.concatenate(all_ocsvm)
    lof_preds       = np.concatenate(all_lof)
    if_preds        = np.concatenate(all_if_pred)
    voting          = np.concatenate(all_voting)
    y_pred          = np.where(ensemble_scores > eff_thr, 1, 0)

    total_time = _time.perf_counter() - t_total_start

  
    total    = len(y_true)
    n_benign = int(np.sum(y_true == 0))
    n_attack = int(np.sum(y_true == 1))

    acc     = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    w_f1    = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    prec, rec, f1c, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], average=None, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn = int(cm[0,0]); fp = int(cm[0,1])
    fn = int(cm[1,0]); tp = int(cm[1,1])

    latency_ms = (inf_time / total) * 1000 if total > 0 else 0
    throughput  = total / inf_time          if inf_time > 0 else 0

    SEP = "=" * 60
    print("\n" + SEP)
    print("  DETECTION RESULTS — Ensemble (IF + LOF + OCSVM)")
    print(SEP)

    print(f"\n  Split Info")
    print(f"  {'─'*40}")
    print(f"  Train (80% benign) : {split_info['n_train_benign']:,} rows")
    print(f"  Test  (20% benign) : {n_benign:,} rows  (unseen)")
    print(f"  Test  (all attacks): {n_attack:,} rows  (unseen)")
    print(f"  Test total         : {total:,} rows")
    print(f"  Threshold          : {eff_thr}")

    print(f"\n  Resource & Performance")
    print(f"  {'─'*40}")
    print(f"  Total Detection Time  : {inf_time:.4f} s")
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

    print(f"  Individual Algorithm Anomaly Detection")
    print(f"  {'─'*40}")
    ocsvm_n = int(np.sum(ocsvm_preds == -1))
    lof_n   = int(np.sum(lof_preds   == -1))
    if_n    = int(np.sum(if_preds    == -1))
    print(f"  One-Class SVM (OCSVM) : {ocsvm_n:>8,}  ({ocsvm_n/total*100:.2f}%)")
    print(f"  Local Outlier Factor  : {lof_n:>8,}  ({lof_n/total*100:.2f}%)")
    print(f"  Isolation Forest      : {if_n:>8,}  ({if_n/total*100:.2f}%)")

    print(f"\n  Voting Consensus Distribution")
    print(f"  {'─'*40}")
    vote_labels = [
        "0 votes  (All say Benign)   ",
        "1 vote   (Likely Benign)    ",
        "2 votes  (Suspicious)       ",
        "3 votes  (High Conf Attack) ",
    ]
    for v in range(4):
        cnt = int(np.sum(voting == v))
        print(f"  {vote_labels[v]}: {cnt:>8,}  ({cnt/total*100:.2f}%)")

    print(f"\n  Ensemble Score Bands")
    print(f"  {'─'*40}")
    high   = int(np.sum(ensemble_scores > 0.75))
    medium = int(np.sum((ensemble_scores > 0.35) & (ensemble_scores <= 0.75)))
    low    = int(np.sum((ensemble_scores > 0.02) & (ensemble_scores <= 0.35)))
    normal = int(np.sum(ensemble_scores <= 0.02))
    print(f"  High   (> 0.75)       : {high:>8,}  ({high/total*100:.2f}%)")
    print(f"  Medium (0.35 - 0.75)  : {medium:>8,}  ({medium/total*100:.2f}%)")
    print(f"  Low    (0.02 - 0.35)  : {low:>8,}  ({low/total*100:.2f}%)")
    print(f"  Normal (<= 0.02)      : {normal:>8,}  ({normal/total*100:.2f}%)")

    print(f"\n  Target Criteria")
    print(f"  {'─'*40}")
    for label, val, thr in [
        ("Attack Recall  >= 98%", rec[1], 0.98),
        ("Benign Recall  >= 85%", rec[0], 0.85),
        ("Weighted F1    >= 98%", w_f1,   0.98),
    ]:
        status = "PASS" if val >= thr else "FAIL"
        print(f"  {label} : {val*100:.2f}%  [{status}]")

    print(f"\n  Total pipeline time   : {total_time:.2f}s")
    print(f"  Log file              : logs/main.log")
    print("=" * 60)

 
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    results = {
        'accuracy': float(acc), 'balanced_accuracy': float(bal_acc),
        'weighted_f1': float(w_f1),
        'benign_precision': float(prec[0]), 'benign_recall': float(rec[0]),
        'attack_precision': float(prec[1]), 'attack_recall': float(rec[1]),
        'attack_f1': float(f1c[1]),
        'confusion_matrix': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp},
        'inference_time_s': round(inf_time, 4),
        'latency_ms': round(latency_ms, 6),
        'throughput': round(throughput, 2),
        'peak_memory_mb': round(peak_mem_mb, 2),
        'avg_cpu_pct': round(avg_cpu_pct, 2),
        'ocsvm_anomalies': ocsvm_n, 'lof_anomalies': lof_n, 'if_anomalies': if_n,
    }
    out = results_dir / f'detection_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved -> {out}")

    db = DatabaseConnector(config)
    db.close()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='IoMT IDS - Baseline Profiling System')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'detect'])
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--sample', type=int, default=None)
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--reset', action='store_true', help='Auto-reset database before training')
    parser.add_argument('--no-reset', action='store_true', help='Skip auto-reset (fail if data exists)')
    
    args = parser.parse_args()
    
    auto_reset = not args.no_reset
    
    try:
        config = load_config(args.config)
        
        logger.info("="*80)
        logger.info("IoMT IDS - BASELINE PROFILING SYSTEM")
        logger.info("="*80)
        logger.info(f"Mode: {args.mode.upper()}")
        logger.info(f"Dataset: {args.dataset}")
        if args.sample:
            logger.info(f"Sample: {args.sample:,}")
        logger.info(f"Auto-reset: {'ON' if auto_reset else 'OFF'}")
        logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if args.mode == 'train':
            train_mode_benign_only(config, args.dataset, args.sample, auto_reset=auto_reset)
        elif args.mode == 'detect':
            detect_mode_with_metrics(config, args.dataset, args.sample)
        
        logger.info("\n✅ [SUCCESS] Operation completed!")
        
    except Exception as e:
        logger.error(f"\n❌ [ERROR] Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()