"""
ULTRA-FAST IDS SIMULATION - DATABASE VERSION
Saves alerts to PostgreSQL + Redis instead of JSONL

Original: Tan Yi Feng (JSONL version)
Modified: Angela Yam Bao Hui (Database integration — Schema V4)

Changes from V3:
- Removed packet_id / alert['id'] usage (column no longer in alerts table)
- Layer 1 individual scores (ocsvm_score, lof_score, isolation_forest_score)
  now passed to connector and saved into featured_traffic
- layer1_pred dict extended with raw scores alongside pred flags
"""

import pandas as pd
import numpy as np
import joblib
import yaml
import sys
import pickle
import logging
import time
import warnings
from pathlib import Path
from datetime import datetime
from collections import deque
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score

warnings.filterwarnings('ignore', message='X does not have valid feature names')


SCRIPT_DIR = Path(__file__).parent
sys.path.append(str(SCRIPT_DIR / 'src'))

from database.db_connector_enhanced import DatabaseConnectorEnhanced



DATASET_PATH = SCRIPT_DIR / 'data' / 'processed' / 'full_inference_cleaned_casing_preserved.csv'
CONFIG_PATH  = SCRIPT_DIR / 'config' / 'config.yaml'
MODELS_DIR   = SCRIPT_DIR / 'data' / 'models'
SRC_PATH     = SCRIPT_DIR / 'src'

sys.path.append(str(SRC_PATH))

from preprocessing.feature_extractor import FeatureExtractor

# Logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

progress_logger = logging.getLogger('progress')
progress_logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(message)s'))
progress_logger.addHandler(console_handler)



ATTACK_LABELS = {
    0:   "Benign (Normal)",
    1:   "ARP Spoofing",
    2:   "MQTT Connect Flood",
    3:   "MQTT Publish Flood",
    4:   "MQTT Malformed",
    5:   "Reconnaissance",
    6:   "Recon (VulnScan)",
    7:   "ICMP Flood",
    8:   "SYN Flood",
    9:   "TCP Flood",
    10:  "UDP Flood",
    100: "Ambiguous"
}

ATTACK_SEVERITY = {
    0:   "INFO",
    1:   "MEDIUM",
    2:   "CRITICAL",
    3:   "CRITICAL",
    4:   "HIGH",
    5:   "MEDIUM",
    6:   "MEDIUM",
    7:   "CRITICAL",
    8:   "CRITICAL",
    9:   "CRITICAL",
    10:  "CRITICAL",
    100: "LOW"
}

ATTACK_DESCRIPTIONS = {
    1:   "ARP cache poisoning detected - potential MITM attack",
    2:   "MQTT connection flood - service disruption attempt",
    3:   "MQTT publish flood - broker overload attack",
    4:   "Malformed MQTT packets - protocol exploitation attempt",
    5:   "Network reconnaissance - information gathering phase",
    6:   "Vulnerability scanning - automated probing detected",
    7:   "ICMP flood - network saturation attack",
    8:   "SYN flood - TCP handshake exhaustion attack",
    9:   "TCP flood - connection-based DoS attack",
    10:  "UDP flood - bandwidth exhaustion attack",
    100: "Uncertain threat - requires manual investigation"
}


def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)




class AlertManagerDB:
    """
    Database-based alert manager using PostgreSQL + Redis.
    Schema V4: writes to featured_traffic first, then alerts.
    packet_id is no longer used — alert_id (PK) is the identifier.
    """

    def __init__(self, config: dict):
        self.config = config
        self.db = DatabaseConnectorEnhanced(config)

        self.alert_counts  = {sev: 0 for sev in set(ATTACK_SEVERITY.values())}
        self.attack_counts = {label: 0 for label in ATTACK_LABELS.keys()}

        progress_logger.info("✅ Database Alert Manager initialized (PostgreSQL + Redis)")

    def save_alert(
        self,
        attack_id: int,
        confidence: float,
        features_dict: dict,
        layer1_predictions: dict = None
    ):
        """
        Save alert to PostgreSQL (Schema V4).

        Args:
            attack_id          : Integer attack type ID (from Layer 2)
            confidence         : Float confidence score 0-1 (from Layer 2)
            features_dict      : Dict of feature names → float values
            layer1_predictions : Dict with Layer 1 data:
                {
                    'ensemble_score'         : 0.85,
                    'voting_consensus'       : 3,
                    'ocsvm_pred'             : -1,   # -1 = anomaly
                    'lof_pred'               : -1,
                    'if_pred'                : -1,
                    'isolation_forest_score' : 0.72, # raw score
                    'lof_score'              : -1.5,
                    'ocsvm_score'            : -0.3
                }

        Note: packet_id removed — no longer stored in alerts table.
        """
        timestamp   = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        severity    = ATTACK_SEVERITY.get(attack_id, "UNKNOWN")
        attack_name = ATTACK_LABELS.get(attack_id, f"Unknown-{attack_id}")
        description = ATTACK_DESCRIPTIONS.get(attack_id, "Unknown threat detected")

        # Update in-memory counters
        self.alert_counts[severity]   = self.alert_counts.get(severity, 0) + 1
        self.attack_counts[attack_id] = self.attack_counts.get(attack_id, 0) + 1

        alert_dict = {
            "timestamp":   timestamp,
            "severity":    severity,
            "attack_type": attack_name,
            "attack_id":   int(attack_id),
            "description": description,
            "confidence":  round(float(confidence), 4),
            "features":    features_dict
        }

        try:
            alert_id = self.db.save_alert_from_jsonl_format(
                alert_dict,
                layer1_predictions=layer1_predictions
            )
            if not alert_id:
                logger.warning("⚠️  Failed to save alert to database")
        except Exception as e:
            logger.error(f"❌ Database save error: {e}")

    def close(self):
        self.db.close()

    def print_summary(self):
        print("\n" + "=" * 80)
        print("📊 ALERT SUMMARY (FROM DATABASE)")
        print("=" * 80)

        stats = self.db.get_alert_statistics()

        print("\nBy Severity Level:")
        for severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'INFO']:
            count = stats.get('severity_counts', {}).get(severity, 0)
            if count > 0:
                print(f"  {severity:>8}: {count:>8,} alerts")

        print("\nBy Attack Type:")
        for attack_type, count in sorted(
            stats.get('attack_counts', {}).items(),
            key=lambda x: x[1],
            reverse=True
        ):
            if count > 0:
                attack_id = next(
                    (k for k, v in ATTACK_LABELS.items() if v == attack_type), 0
                )
                sev = ATTACK_SEVERITY.get(attack_id, "UNKNOWN")
                print(f"  [{sev:>8}] {attack_type:<25}: {count:>8,}")

        print(f"\n💾 Total alerts in database: {stats.get('total_alerts', 0):,}")
        avg_conf = stats.get('avg_confidence') or 0
        print(f"📊 Average confidence: {avg_conf:.2%}")

        cache_stats = self.db.get_cache_stats()
        print(f"\n🔄 Redis Cache Statistics:")
        print(f"  Cache hits:   {cache_stats['hits']:,}")
        print(f"  Cache misses: {cache_stats['misses']:,}")
        print(f"  Hit rate:     {cache_stats['hit_rate']:.2f}%")



class RealTimeIDS:

    def __init__(self, models_dir, config):
        self.config = config
        self.alert_manager = AlertManagerDB(config)

        progress_logger.info("Loading IDS models...")
        self.ensemble      = joblib.load(models_dir / 'ensemble.joblib')
        self.scaler        = joblib.load(models_dir / 'scaler.joblib')
        self.rf_pipeline   = joblib.load(models_dir / 'best_rf_pipeline2.pkl')

        with open(models_dir / 'selected_features2.pkl', 'rb') as f:
            self.selected_features = list(pickle.load(f))

        self.feature_extractor = FeatureExtractor()
        self.threshold = config.get('ensemble', {}).get('threshold_low', 0.1)

        # Stats
        self.total_packets      = 0
        self.layer1_suspicious  = 0
        self.layer2_attacks     = 0
        self.processing_times   = deque(maxlen=100)

        # Accuracy tracking
        self.all_y_true = []
        self.all_y_pred = []

        progress_logger.info("Models loaded successfully")

    def process_batch(self, batch_df, batch_num, total_batches):
        """
        Process batch with Layer 1 + Layer 2 detection.
        Saves full Layer 1 scores (not just anomaly flags) into featured_traffic.
        """
        batch_start = time.time()
        batch_size  = len(batch_df)

        progress_logger.info(
            f"[Batch {batch_num}/{total_batches}] Processing {batch_size} packets..."
        )

       

        features_df  = self.feature_extractor.extract_from_dataframe(batch_df)
        features_norm, _ = self.feature_extractor.normalize_features(features_df, self.scaler)
        X_L1         = self.feature_extractor.get_feature_vector(features_norm)

        predictions  = self.ensemble.predict_with_details(X_L1)

        layer1_flags      = predictions['ensemble_score'] > self.threshold
        suspicious_indices = np.where(layer1_flags)[0]

        self.total_packets     += batch_size
        self.layer1_suspicious += len(suspicious_indices)

        batch_predictions = np.zeros(batch_size, dtype=int)

      

        if len(suspicious_indices) > 0:
            try:
                X_L2_subset = (
                    batch_df.iloc[suspicious_indices][self.selected_features]
                    .copy()
                )
                X_L2_subset.replace([np.inf, -np.inf], np.nan, inplace=True)

                imputer     = SimpleImputer(strategy='median')
                X_L2_clean  = imputer.fit_transform(X_L2_subset)

                probs       = self.rf_pipeline.predict_proba(X_L2_clean)
                max_probs   = np.max(probs, axis=1)
                raw_preds   = np.argmax(probs, axis=1)

                mapped_preds  = raw_preds + 1
                l2_decisions  = np.where(max_probs >= 0.9, mapped_preds, 100)

                batch_predictions[suspicious_indices] = l2_decisions

               

                for i, (sus_idx, attack_id, confidence) in enumerate(
                    zip(suspicious_indices, l2_decisions, max_probs)
                ):
                    if attack_id != 0:  
                       
                        layer1_pred = {
                            'ensemble_score':         float(predictions['ensemble_score'][sus_idx]),
                            'voting_consensus':       int(predictions['voting_consensus'][sus_idx]),
                           
                            'isolation_forest_score': float(predictions.get('if_score',   [0])[sus_idx])
                                                      if 'if_score'   in predictions else None,
                            'lof_score':              float(predictions.get('lof_score',  [0])[sus_idx])
                                                      if 'lof_score'  in predictions else None,
                            'ocsvm_score':            float(predictions.get('ocsvm_score',[0])[sus_idx])
                                                      if 'ocsvm_score' in predictions else None,
                         
                            'ocsvm_pred': int(predictions['ocsvm_pred'][sus_idx]),
                            'lof_pred':   int(predictions['lof_pred'][sus_idx]),
                            'if_pred':    int(predictions['if_pred'][sus_idx]),
                        }

                       
                        feature_dict = {}
                        packet_row   = batch_df.iloc[sus_idx]
                        for feat_name in self.selected_features:
                            if feat_name in packet_row.index:
                                feat_value = packet_row[feat_name]
                                if pd.notna(feat_value) and not np.isinf(feat_value):
                                    feature_dict[feat_name] = float(feat_value)

                        
                        self.alert_manager.save_alert(
                            attack_id=attack_id,
                            confidence=confidence,
                            features_dict=feature_dict,
                            layer1_predictions=layer1_pred
                        )

                        self.layer2_attacks += 1

            except Exception as e:
                logger.error(f"Layer 2 error: {e}")

       

        if 'label' in batch_df.columns:
            batch_y_true = (
                batch_df['label'].astype(str)
                .str.split('.').str[0]
                .astype(int).values
            )
            self.all_y_true.extend(batch_y_true)
            self.all_y_pred.extend(batch_predictions)

        batch_time = time.time() - batch_start
        self.processing_times.append(batch_time)

        progress_logger.info(
            f"[Batch {batch_num}/{total_batches}] Complete in {batch_time:.2f}s | "
            f"Suspicious: {len(suspicious_indices)} | "
            f"Total Attacks Saved: {self.layer2_attacks:,}\n"
        )
        return batch_predictions, batch_time

    def get_stats(self):
        avg_time = np.mean(self.processing_times) if self.processing_times else 0
        return {
            'total_packets':    self.total_packets,
            'layer1_suspicious': self.layer1_suspicious,
            'layer2_attacks':   self.layer2_attacks,
            'avg_batch_time':   avg_time,
            'avg_latency':      (avg_time / 500) if avg_time > 0 else 0
        }

    def print_accuracy_report(self):
        if not self.all_y_true:
            print("\nNo labels found. Skipping accuracy report.")
            return

        y_true = np.array(self.all_y_true)
        y_pred = np.array(self.all_y_pred)

        print("\n" + "=" * 80)
        print("🎯 ACCURACY VERIFICATION (HIGH CONFIDENCE ONLY)")
        print("=" * 80)

        ambiguous_count = np.sum(y_pred == 100)
        total = len(y_pred)
        print(f"Ambiguous Skipped (Label 100): {ambiguous_count:,} "
              f"({ambiguous_count / total * 100:.2f}%)")

        mask           = y_pred != 100
        y_true_certain = y_true[mask]
        y_pred_certain = y_pred[mask]

        if len(y_true_certain) > 0:
            acc = accuracy_score(y_true_certain, y_pred_certain)
            print(f"Accuracy (Certain): {acc:.4f}")
            print("\nDetailed Classification Report (High Confidence Only):")

            unique_labels  = np.unique(
                np.concatenate((y_true_certain, y_pred_certain))
            )
            labels_to_show = [l for l in unique_labels if l != 0 and l != 100]
            target_names   = [
                ATTACK_LABELS.get(l, f"Class {l}") for l in labels_to_show
            ]

            if labels_to_show:
                print(classification_report(
                    y_true_certain,
                    y_pred_certain,
                    labels=labels_to_show,
                    target_names=target_names,
                    digits=4,
                    zero_division=0
                ))
        else:
            print("   No confident predictions were made.")

    def cleanup(self):
        self.alert_manager.close()



def run_simulation(dataset_path, packets_per_second=300, duration=None):
    print("🚀 STARTING ULTRA-FAST IDS SIMULATION (DATABASE VERSION)")
    print("📦 Saving to: PostgreSQL + Redis (not JSONL)")
    print("=" * 80)

    config = load_config(CONFIG_PATH)
    ids    = RealTimeIDS(MODELS_DIR, config)

    try:
        progress_logger.info(f"Loading dataset: {dataset_path}")
        df             = pd.read_csv(dataset_path)
        total_packets  = len(df)

        batch_size = packets_per_second
        num_batches = (total_packets + batch_size - 1) // batch_size
        if duration:
            num_batches = min(num_batches, duration)

        print(f"Simulation Parameters:")
        print(f"  Batch Size:    {batch_size:,} packets/batch")
        print(f"  Total Batches: {num_batches:,}")
        print("=" * 80 + "\n")

        start_time = time.time()

        for batch_num in range(1, num_batches + 1):
            start_idx = (batch_num - 1) * batch_size
            end_idx   = min(start_idx + batch_size, total_packets)
            batch_df  = df.iloc[start_idx:end_idx].copy()

            ids.process_batch(batch_df, batch_num, num_batches)
            time.sleep(0.05)

        total_time = time.time() - start_time
        stats      = ids.get_stats()

        print("\n" + "=" * 80)
        print("🏁 SIMULATION COMPLETE")
        print("=" * 80)
        print(f"\nProcessing Statistics:")
        print(f"  Total Packets:  {stats['total_packets']:,}")
        print(f"  Throughput:     {stats['total_packets'] / total_time:.2f} packets/s")
        print(f"  Avg Latency:    {stats['avg_latency'] * 1000:.2f}ms")

        ids.alert_manager.print_summary()
        ids.print_accuracy_report()

        print("\n✅ Simulation finished successfully")
        print("✅ All alerts saved to PostgreSQL database")
        print("✅ Redis cache statistics shown above")

    finally:
        ids.cleanup()


if __name__ == "__main__":
    run_simulation(
        dataset_path=DATASET_PATH,
        packets_per_second=500,
        duration=5,  
    )