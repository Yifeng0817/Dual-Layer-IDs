"""
ENHANCED: Preserve scenario column in cleaned data
This allows device-specific profiling!

Author: Angela Yam Bao Hui
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def detect_contamination(df: pd.DataFrame) -> dict:
    """Detect mislabeled data (attack scenarios labeled as benign)"""
    if 'label' not in df.columns or 'scenario' not in df.columns:
        return {'contaminated': 0, 'total': 0}
    
    df_temp = df.copy()
    df_temp['label'] = df_temp['label'].astype(str).str.strip().str.lower()
    df_temp['label'] = df_temp['label'].replace({
        'normal': '0.0',
        'benign': '0.0',
        'background': '0.0'
    })
    
    benign_mask = df_temp['label'] == '0.0'
    benign_df = df_temp[benign_mask]
    
    if len(benign_df) == 0:
        return {'contaminated': 0, 'total': 0, 'details': []}
    
    attack_keywords = [
        'ddos', 'dos', 'attack', 'mitm', 'scan', 'exploit',
        'malware', 'intrusion', 'breach', 'flood', 'syn_flood',
        'udp_flood', 'icmp_flood', 'reconnaissance', 'brute'
    ]
    
    contaminated_count = 0
    contamination_details = []
    
    for keyword in attack_keywords:
        mask = benign_df['scenario'].astype(str).str.lower().str.contains(keyword, na=False)
        keyword_count = mask.sum()
        
        if keyword_count > 0:
            contaminated_count += keyword_count
            contamination_details.append({
                'keyword': keyword,
                'count': keyword_count,
                'percent': keyword_count / len(benign_df) * 100
            })
    
    return {
        'contaminated': contaminated_count,
        'total': len(benign_df),
        'percent': contaminated_count / len(benign_df) * 100 if len(benign_df) > 0 else 0,
        'details': contamination_details
    }


def preprocess_dataset(input_path: str, output_path: str, sample_benign: int = None, auto_clean: bool = True):
    """
    ✅ ENHANCED: Preserves scenario column for device profiling
    """
    
    logger.info("="*80)
    logger.info("ENHANCED PREPROCESSING (Scenario Preservation)")
    logger.info("="*80)
    
   
    logger.info(f"\n[STEP 1] Loading dataset from {input_path}...")
    df = pd.read_csv(input_path, low_memory=False)
    logger.info(f"Loaded {len(df):,} records")
    
    
    logger.info("\n[STEP 2] Cleaning data...")
    df = df.dropna(how='all')
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df[numeric_cols] = df[numeric_cols].fillna(0)
    logger.info("Cleaned infinite and NaN values")
    
  
    logger.info("\n[STEP 3] Removing duplicates...")
    before_dup = len(df)
    df = df.drop_duplicates()
    logger.info(f"Removed {before_dup - len(df):,} duplicates")
    
   
    logger.info("\n[STEP 4] Preserving scenario column...")
    
    if 'scenario' in df.columns:
        
        df['scenario'] = df['scenario'].fillna('benign_unknown')
        logger.info(f"✅ Scenario column preserved")
        logger.info(f"   Unique scenarios: {df['scenario'].nunique()}")
    else:
        logger.warning("⚠️  No scenario column in dataset!")
    
   
    logger.info("\n[STEP 5] Consolidating label columns...")
    
    label_cols = [col for col in df.columns if col.strip().lower() == 'label']
    
    if len(label_cols) > 1:
        df['label_consolidated'] = df[label_cols[0]]
        for col in label_cols[1:]:
            df['label_consolidated'] = df['label_consolidated'].fillna(df[col])
        df = df.drop(columns=label_cols)
        df = df.rename(columns={'label_consolidated': 'label'})
        logger.info("✅ Consolidated label columns")
    
   
    logger.info("\n[STEP 6] ⚠️  DETECTING CONTAMINATION...")
    
    contamination = detect_contamination(df)
    
    if contamination['contaminated'] > 0:
        logger.warning("="*80)
        logger.warning("❌ CONTAMINATION DETECTED!")
        logger.warning("="*80)
        logger.warning(f"Found {contamination['contaminated']:,} contaminated benign samples")
        logger.warning(f"({contamination['percent']:.1f}% of benign data)")
        
        for detail in contamination['details']:
            logger.warning(f"  - '{detail['keyword']}': {detail['count']:,} samples")
        
        if auto_clean:
            logger.info("\n🔧 AUTO-CLEANING...")
            
            df['label'] = df['label'].astype(str).str.strip().str.lower()
            df['label'] = df['label'].replace({
                'normal': '0.0',
                'benign': '0.0',
                'background': '0.0'
            })
            
            benign_before = (df['label'] == '0.0').sum()
            
           
            attack_keywords = [
                'ddos', 'dos', 'attack', 'mitm', 'scan', 'exploit',
                'malware', 'intrusion', 'breach', 'flood'
            ]
            
            contaminated_mask = pd.Series([False] * len(df), index=df.index)
            benign_mask = df['label'] == '0.0'
            
            for keyword in attack_keywords:
                keyword_mask = df['scenario'].astype(str).str.lower().str.contains(keyword, na=False)
                contaminated_mask = contaminated_mask | (benign_mask & keyword_mask)
            
          
            df = df[~contaminated_mask].copy()
            
            benign_after = (df['label'] == '0.0').sum()
            
            logger.info(f"✅ Removed {benign_before - benign_after:,} contaminated samples")
            logger.info(f"   Clean benign: {benign_after:,}")
            
            
            benign_mask_after = df['label'] == '0.0'
            df.loc[benign_mask_after, 'scenario'] = 'benign_normal'
            logger.info(f"✅ All clean benign labeled as 'benign_normal' scenario")
    
   
    logger.info("\n[STEP 7] Standardizing labels...")
    
    if 'label' in df.columns:
        df['label'] = df['label'].astype(str).str.strip().str.lower()
        df['label'] = df['label'].replace({
            'normal': '0.0',
            'benign': '0.0',
            'background': '0.0',
            'nan': 'unknown',
            'none': 'unknown',
            '': 'unknown'
        })
        
        benign_df = df[df['label'] == '0.0'].copy()
        attack_df = df[df['label'] != '0.0'].copy()
        
        if 'unknown' in attack_df['label'].values:
            unknown_count = (attack_df['label'] == 'unknown').sum()
            logger.info(f"Removing {unknown_count:,} unknown labels")
            attack_df = attack_df[attack_df['label'] != 'unknown']
        
        if sample_benign and len(benign_df) > sample_benign:
            logger.info(f"\n[STEP 8] Downsampling benign to {sample_benign:,}")
            benign_df = benign_df.sample(n=sample_benign, random_state=42)
        
        df = pd.concat([benign_df, attack_df], ignore_index=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        benign_final = (df['label'] == '0.0').sum()
        attack_final = (df['label'] != '0.0').sum()
        
        logger.info(f"\nFinal distribution:")
        logger.info(f"  Benign:  {benign_final:,} ({benign_final/len(df)*100:.2f}%)")
        logger.info(f"  Attacks: {attack_final:,} ({attack_final/len(df)*100:.2f}%)")
    
    logger.info("\n[STEP 7.5] Assigning scenario to benign samples...")

    if 'scenario' in df.columns:
        
        benign_mask = df['label'] == '0.0'
        null_scenario_benign = df[benign_mask]['scenario'].isna().sum()
        
        if null_scenario_benign > 0:
            logger.info(f"   Found {null_scenario_benign:,} benign samples with NULL scenario")
            df.loc[benign_mask & df['scenario'].isna(), 'scenario'] = 'benign_normal'
            logger.info(f"   ✅ Assigned them to scenario: 'benign_normal'")
    else:
    
        logger.info("   Creating scenario column...")
        benign_mask = df['label'] == '0.0'
        df['scenario'] = 'unknown'
        df.loc[benign_mask, 'scenario'] = 'benign_normal'
        logger.info(f"   ✅ Created scenario column")
        logger.info(f"      - Benign: 'benign_normal'")
        logger.info(f"      - Attack: 'unknown' (will be filled if available)")

    
    benign_scenarios = df[df['label'] == '0.0']['scenario'].unique()
    logger.info(f"\n   Final benign scenarios: {benign_scenarios}")
   
    logger.info("\n[STEP 9] Verifying scenario preservation...")
    
    if 'scenario' in df.columns:
        benign_scenarios = df[df['label'] == '0.0']['scenario'].nunique()
        attack_scenarios = df[df['label'] != '0.0']['scenario'].nunique()
        
        logger.info(f"  ✅ Benign scenarios: {benign_scenarios}")
        logger.info(f"  ✅ Attack scenarios: {attack_scenarios}")
        logger.info(f"  ✅ Total scenarios: {df['scenario'].nunique()}")
    else:
        logger.warning("  ⚠️  No scenario column!")
    
  
    logger.info(f"\n[STEP 10] Saving to {output_path}...")
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    logger.info(f"Saved {len(df):,} records")
    logger.info(f"File size: {Path(output_path).stat().st_size / 1024 / 1024:.2f} MB")
    
    logger.info("\n" + "="*80)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("="*80)
    
    return df


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced preprocessing with scenario preservation')
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--sample-benign', type=int, default=None)
    parser.add_argument('--no-auto-clean', action='store_true')
    
    args = parser.parse_args()
    
    try:
        df = preprocess_dataset(
            args.input, 
            args.output, 
            args.sample_benign,
            auto_clean=not args.no_auto_clean
        )
        
        logger.info("\n[SUCCESS] Preprocessing complete!")
        logger.info(f"Next step: python main.py --mode train --dataset {args.output}")
        
    except Exception as e:
        logger.error(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)