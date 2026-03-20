"""
Real-Time IDS Alert Dashboard - DATABASE VERSION 
Reads from PostgreSQL + Redis instead of JSONL

Original: Tan Yi Feng (JSONL version)
Modified: Angela Yam Bao Hui (Database integration + UI redesign v2.1)
"""

import streamlit as st
import pandas as pd
import time
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import numpy as np



sys.path.append(str(Path(__file__).parent / 'src'))

from database.db_connector_enhanced import DatabaseConnectorEnhanced
import yaml


st.set_page_config(
    page_title="IDS Alert Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

if 'alert_statuses' not in st.session_state:
    st.session_state.alert_statuses = {}
if 'selected_alert_id' not in st.session_state:
    st.session_state.selected_alert_id = None
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Dashboard"
# ── FIX #1: auto-refresh lives in session_state so it's always accessible ──
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = True
if 'refresh_interval' not in st.session_state:
    st.session_state.refresh_interval = 5

st.markdown("""
<style>
    /* ─── Reset & Global ─── */
    .block-container {
        padding-top: 5.5rem;
        padding-bottom: 1.5rem;
        max-width: 1400px;
    }
    h1, h2, h3, h4, h5, h6 {
        letter-spacing: -0.01em;
    }
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 0.5rem;
    }

    /* ─── Sidebar ─── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0e1a 0%, #0d1225 100%);
        border-right: 1px solid rgba(99,102,241,0.15);
    }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stMultiSelect label,
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stCheckbox label {
        font-size: 0.82rem !important;
        color: #8b8fa3 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* ─── Scrollbar — dark rounded ─── */
    ::-webkit-scrollbar {
        width: 7px;
        height: 7px;
    }
    ::-webkit-scrollbar-track {
        background: #0a0a0a;
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb {
        background: #1e1e1e;
        border-radius: 10px;
        border: 1px solid #111111;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #2a2a2a;
    }
    ::-webkit-scrollbar-corner {
        background: #0a0a0a;
    }
    /* Firefox */
    * {
        scrollbar-width: thin;
        scrollbar-color: #1e1e1e #0a0a0a;
    }

    /* ─── Metric Cards (Dashboard summary row) ─── */
    .metric-strip {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        gap: 12px;
        margin-bottom: 24px;
    }
    .mc {
        background: linear-gradient(145deg, #111827 0%, #0f172a 100%);
        border: 1px solid #1e293b;
        border-radius: 14px;
        padding: 18px 16px;
        text-align: center;
        position: relative;
        overflow: hidden;
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .mc:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 32px rgba(0,0,0,0.35);
    }
    .mc::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        border-radius: 14px 14px 0 0;
    }
    .mc-total::before  { background: linear-gradient(90deg, #8b5cf6, #6366f1); }
    .mc-total:hover    { border-color: #6366f1; }
    .mc-crit::before   { background: linear-gradient(90deg, #ef4444, #dc2626); }
    .mc-crit:hover     { border-color: #ef4444; }
    .mc-high::before   { background: linear-gradient(90deg, #f97316, #ea580c); }
    .mc-high:hover     { border-color: #f97316; }
    .mc-med::before    { background: linear-gradient(90deg, #eab308, #ca8a04); }
    .mc-med:hover      { border-color: #eab308; }
    .mc-low::before    { background: linear-gradient(90deg, #3b82f6, #2563eb); }
    .mc-low:hover      { border-color: #3b82f6; }
    .mc-amb::before    { background: linear-gradient(90deg, #06b6d4, #0891b2); }
    .mc-amb:hover      { border-color: #06b6d4; }
    .mc-conf::before   { background: linear-gradient(90deg, #10b981, #059669); }
    .mc-conf:hover     { border-color: #10b981; }
    .mc .mc-icon  { font-size: 1.4rem; margin-bottom: 2px; }
    .mc .mc-val   { font-size: 2rem; font-weight: 800; line-height: 1.1; }
    .mc .mc-label { font-size: 0.7rem; color: #64748b; text-transform: uppercase;
                    letter-spacing: 1.2px; margin-top: 4px; }
    .mc .mc-pct   { font-size: 0.72rem; color: #475569; margin-top: 2px; }
    .mc-total .mc-val { color: #a78bfa; }
    .mc-crit  .mc-val { color: #ef4444; }
    .mc-high  .mc-val { color: #f97316; }
    .mc-med   .mc-val { color: #eab308; }
    .mc-low   .mc-val { color: #3b82f6; }
    .mc-amb   .mc-val { color: #06b6d4; }
    .mc-conf  .mc-val { color: #10b981; }

    /* ─── Severity Badges ─── */
    .sev-badge {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        vertical-align: middle;
    }
    .sev-critical { background: rgba(239,68,68,0.12); color: #f87171;
                    border: 1.5px solid rgba(239,68,68,0.4); }
    .sev-high     { background: rgba(249,115,22,0.12); color: #fb923c;
                    border: 1.5px solid rgba(249,115,22,0.4); }
    .sev-medium   { background: rgba(234,179,8,0.12);  color: #facc15;
                    border: 1.5px solid rgba(234,179,8,0.4); }
    .sev-low      { background: rgba(59,130,246,0.12);  color: #60a5fa;
                    border: 1.5px solid rgba(59,130,246,0.4); }
    .sev-info     { background: rgba(16,185,129,0.12);  color: #34d399;
                    border: 1.5px solid rgba(16,185,129,0.4); }

    /* ─── Status Badges ─── */
    .stat-badge {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        vertical-align: middle;
    }
    .stat-new            { background: rgba(139,92,246,0.12); color: #a78bfa;
                           border: 1.5px solid rgba(139,92,246,0.4); }
    .stat-reviewed       { background: rgba(59,130,246,0.12); color: #60a5fa;
                           border: 1.5px solid rgba(59,130,246,0.4); }
    .stat-investigating  { background: rgba(245,158,11,0.12); color: #fbbf24;
                           border: 1.5px solid rgba(245,158,11,0.4); }
    .stat-resolved       { background: rgba(16,185,129,0.12); color: #34d399;
                           border: 1.5px solid rgba(16,185,129,0.4); }
    .stat-false-positive { background: rgba(107,114,128,0.12); color: #9ca3af;
                           border: 1.5px solid rgba(107,114,128,0.4); }

    /* ─── Investigation Card (alert list items) ─── */
    .inv-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 8px;
        cursor: pointer;
        transition: all 0.15s ease;
        display: grid;
        grid-template-columns: 5px 1fr auto;
        gap: 14px;
        align-items: center;
    }
    .inv-card:hover {
        border-color: #6366f1;
        background: #131b32;
        box-shadow: 0 4px 16px rgba(99,102,241,0.08);
    }
    .inv-card.selected {
        border-color: #818cf8;
        background: #1a1f3d;
        box-shadow: 0 0 0 1px rgba(129,140,248,0.3),
                    0 4px 16px rgba(99,102,241,0.12);
    }
    .inv-indicator {
        width: 5px;
        height: 40px;
        border-radius: 4px;
    }
    .ind-critical { background: #ef4444; box-shadow: 0 0 8px rgba(239,68,68,0.4); }
    .ind-high     { background: #f97316; box-shadow: 0 0 8px rgba(249,115,22,0.3); }
    .ind-medium   { background: #eab308; }
    .ind-low      { background: #3b82f6; }
    .ind-info     { background: #10b981; }
    .inv-body .inv-title { font-weight: 600; font-size: 0.92rem; color: #e2e8f0; }
    .inv-body .inv-sub   { font-size: 0.76rem; color: #64748b; margin-top: 2px; }
    .inv-right { text-align: right; }
    .inv-right .inv-time { font-size: 0.76rem; color: #64748b; }
    .inv-right .inv-conf { font-size: 0.88rem; font-weight: 700; }

    /* ─── Detail Panel (right side) ─── */
    .detail-panel {
        background: linear-gradient(145deg, #0f172a 0%, #111827 100%);
        border: 1px solid #1e293b;
        border-radius: 16px;
        padding: 28px;
        position: relative;
    }
    .detail-panel .dp-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 20px;
        padding-bottom: 16px;
        border-bottom: 1px solid #1e293b;
    }
    .detail-panel .dp-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #f1f5f9;
    }
    .detail-panel .dp-id {
        font-size: 0.8rem;
        color: #475569;
        font-family: 'SF Mono', 'Fira Code', monospace;
        background: rgba(30,41,59,0.6);
        padding: 4px 10px;
        border-radius: 8px;
    }
    .dp-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 14px;
        margin-bottom: 20px;
    }
    .dp-field {
        background: rgba(15,23,42,0.5);
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 12px 14px;
    }
    .dp-field .dp-flabel {
        font-size: 0.72rem; color: #64748b;
        text-transform: uppercase; letter-spacing: 0.5px;
    }
    .dp-field .dp-fvalue {
        font-size: 0.95rem; color: #e2e8f0;
        font-weight: 600; margin-top: 2px;
    }

    /* ─── Confidence Gauge ─── */
    .conf-bar-outer {
        background: #1e293b;
        border-radius: 10px;
        height: 10px;
        overflow: hidden;
        margin: 10px 0 4px 0;
    }
    .conf-bar-inner {
        height: 100%;
        border-radius: 10px;
        transition: width 0.6s ease;
    }

    /* ─── Empty State ─── */
    .empty-state {
        text-align: center;
        padding: 80px 20px;
    }
    .empty-state .es-icon {
        font-size: 4.5rem;
        margin-bottom: 16px;
        filter: drop-shadow(0 0 20px rgba(16,185,129,0.3));
    }
    .empty-state .es-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #10b981;
        margin-bottom: 6px;
    }
    .empty-state .es-sub {
        font-size: 0.9rem;
        color: #64748b;
        max-width: 400px;
        margin: 0 auto;
    }

    /* ─── Select-alert placeholder ─── */
    .select-placeholder {
        text-align: center;
        padding: 80px 20px;
        border: 2px dashed #1e293b;
        border-radius: 16px;
    }
    .select-placeholder .sp-icon  { font-size: 3.5rem; margin-bottom: 12px; opacity: 0.4; }
    .select-placeholder .sp-title { font-size: 1.05rem; font-weight: 600; color: #475569; }
    .select-placeholder .sp-sub   { font-size: 0.82rem; color: #334155; margin-top: 4px; }

    /* ─── ML Cards ─── */
    .ml-card {
        background: linear-gradient(145deg, #0f172a 0%, #1e1b4b 100%);
        border: 1px solid rgba(99,102,241,0.2);
        border-radius: 14px;
        padding: 20px;
        text-align: center;
        transition: transform 0.2s ease;
    }
    .ml-card:hover { transform: translateY(-2px); }
    .ml-card .ml-icon  { font-size: 1.5rem; margin-bottom: 4px; }
    .ml-card .ml-val   { font-size: 1.9rem; font-weight: 800; color: #a78bfa; }
    .ml-card .ml-label { font-size: 0.7rem; color: #7c7fa8;
                         text-transform: uppercase; letter-spacing: 1px; margin-top: 2px; }
    .ml-card .ml-pct   { font-size: 0.78rem; color: #6366f1; margin-top: 4px; }

    /* ─── Section Header ─── */
    .sec-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 24px 0 14px 0;
        padding-bottom: 10px;
        border-bottom: 1px solid #1e293b;
    }
    .sec-header h3 {
        font-size: 1.05rem; font-weight: 700; margin: 0; color: #e2e8f0;
    }
    .sec-header .sec-count {
        background: rgba(99,102,241,0.15);
        color: #818cf8;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.72rem;
        font-weight: 600;
    }

    /* ─── Page header ─── */
    .page-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 8px 0 16px 0;
        border-bottom: 1px solid #1e293b;
        margin-bottom: 24px;
    }
    .page-header .ph-left {
        display: flex; align-items: center; gap: 12px;
    }
    .page-header .ph-icon  { font-size: 1.8rem; }
    .page-header .ph-title { font-size: 1.5rem; font-weight: 800;
                             color: #f1f5f9; letter-spacing: -0.02em; }
    .page-header .ph-sub   { font-size: 0.8rem; color: #64748b; }
    .page-header .ph-right { text-align: right; }
    .page-header .ph-time  { font-size: 0.78rem; color: #475569; }
    .page-header .ph-status {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .ph-live {
        background: rgba(16,185,129,0.12);
        color: #34d399;
        border: 1px solid rgba(16,185,129,0.3);
        animation: pulse-live 2s infinite;
    }
    .ph-paused {
        background: rgba(100,116,139,0.12);
        color: #94a3b8;
        border: 1px solid rgba(100,116,139,0.3);
    }
    @keyframes pulse-live {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }

    /* ─── Baseline info-box ─── */
    .baseline-info {
        background: rgba(15,23,42,0.7);
        border: 1px solid #1e293b;
        border-left: 3px solid #6366f1;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 16px;
        font-size: 0.82rem;
        color: #94a3b8;
        line-height: 1.6;
    }
    .baseline-info strong { color: #e2e8f0; }

    /* ─── Baseline overlay legend ─── */
    .baseline-legend {
        display: flex;
        gap: 18px;
        align-items: center;
        padding: 10px 16px;
        background: rgba(15,23,42,0.6);
        border: 1px solid #1e293b;
        border-radius: 10px;
        margin-bottom: 12px;
    }
    .baseline-legend .bl-item {
        display: flex; align-items: center; gap: 6px;
        font-size: 0.8rem; color: #94a3b8;
    }
    .baseline-legend .bl-dot {
        width: 10px; height: 10px; border-radius: 50%; display: inline-block;
    }

    /* ─── Deviation gauge ─── */
    .dev-gauge {
        background: rgba(15,23,42,0.6);
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        margin-top: 12px;
    }
    .dev-gauge .dg-value {
        font-size: 2rem; font-weight: 800; line-height: 1.2;
    }
    .dev-gauge .dg-label {
        font-size: 0.7rem; color: #64748b;
        text-transform: uppercase; letter-spacing: 1px; margin-top: 4px;
    }
    .dev-gauge .dg-verdict {
        font-size: 0.8rem; font-weight: 600; margin-top: 6px;
        padding: 3px 12px; border-radius: 20px; display: inline-block;
    }
    .dg-normal  { color: #34d399; background: rgba(16,185,129,0.12);
                  border: 1px solid rgba(16,185,129,0.3); }
    .dg-warning { color: #fbbf24; background: rgba(245,158,11,0.12);
                  border: 1px solid rgba(245,158,11,0.3); }
    .dg-danger  { color: #f87171; background: rgba(239,68,68,0.12);
                  border: 1px solid rgba(239,68,68,0.3); }

    /* ─── Footer ─── */
    .dash-footer {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 12px 0;
        margin-top: 32px;
        border-top: 1px solid #1e293b;
        font-size: 0.75rem;
        color: #475569;
    }

    /* ─── Expander tweaks ─── */
    .streamlit-expanderHeader { font-size: 0.85rem !important; }

    /* ─── Tabs ─── */
    .stTabs [data-baseweb="tab-list"] { gap: 2px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


ATTACK_COLORS = {
    'ARP Spoofing': '#FFA500', 'MQTT Connect Flood': '#FF0000',
    'MQTT Publish Flood': '#DC143C', 'MQTT Malformed': '#FF6347',
    'Reconnaissance': '#FFD700', 'Recon (VulnScan)': '#DAA520',
    'ICMP Flood': '#8B0000', 'SYN Flood': '#B22222',
    'TCP Flood': '#CD5C5C', 'UDP Flood': '#F08080',
    'Ambiguous': '#00BFFF'
}

SEVERITY_COLORS = {
    'CRITICAL': '#ef4444', 'HIGH': '#f97316',
    'MEDIUM': '#eab308', 'LOW': '#3b82f6', 'INFO': '#10b981'
}

SEVERITY_ORDER = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'INFO']

ALERT_STATUS_OPTIONS = ['NEW', 'REVIEWED', 'INVESTIGATING', 'RESOLVED']
AMBIGUOUS_STATUS_OPTIONS = [
    'NEW', 'REVIEWED', 'INVESTIGATING', 'RESOLVED', 'FALSE POSITIVE'
]

PAGES = ["Dashboard", "Alerts", "ML Analytics", "Feature Analysis"]
PAGE_ICONS = {
    "Dashboard": "📊", "Alerts": "🚨",
    "ML Analytics": "🔬", "Feature Analysis": "🧬"
}
PAGE_SUBTITLES = {
    "Dashboard": "System overview & threat summary",
    "Alerts": "Browse, investigate & acknowledge",
    "ML Analytics": "Layer 1 & Layer 2 model insights",
    "Feature Analysis": "Feature distributions & baseline comparison"
}


@st.cache_resource
def get_database_connection():
    config_path = Path(__file__).parent / 'config' / 'config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return DatabaseConnectorEnhanced(config)


@st.cache_data(ttl=1)
def load_alerts_from_db(limit=1000, severity_filter=None, min_confidence=0.0):
    db = get_database_connection()
    try:
        alerts = db.get_recent_alerts_for_dashboard(
            limit=limit, severity_filter=severity_filter,
            min_confidence=min_confidence
        )
        if not alerts:
            return pd.DataFrame()
        df = pd.DataFrame(alerts)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception as e:
        st.error(f"❌ Database error: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=1)
def load_layer1_stats():
    db = get_database_connection()
    try:
        return db.get_layer1_statistics()
    except Exception as e:
        st.error(f"❌ Layer 1 stats error: {e}")
        return {}


@st.cache_data(ttl=60)
def load_device_baselines():
    """Fetch learned baselines for overlay comparison."""
    db = get_database_connection()
    try:
        if hasattr(db, 'get_device_baselines'):
            return db.get_device_baselines()
    except Exception:
        pass
    return {}


def severity_emoji(sev):
    return {'CRITICAL': '🔴', 'HIGH': '🟠', 'MEDIUM': '🟡',
            'LOW': '🔵', 'INFO': '🟢'}.get(sev, '⚪')

def severity_css(sev):
    return f"sev-{sev.lower()}" if sev else "sev-info"

def status_css(status):
    return f"stat-{status.lower().replace(' ', '-')}"

def confidence_color(conf):
    if conf >= 0.95: return '#10b981'
    if conf >= 0.85: return '#3b82f6'
    if conf >= 0.70: return '#eab308'
    return '#ef4444'

def get_alert_status(alert_id):
    """Read alert status from DATABASE (persists across refreshes)"""
    try:
        db = get_database_connection()
        return db.get_alert_status_from_db(alert_id)
    except Exception as e:
       
        return st.session_state.alert_statuses.get(alert_id, {
            'status': 'NEW', 'analyst': '', 'notes': '', 'updated_at': None
        })

def set_alert_status(alert_id, status, analyst='', notes=''):
    """Write alert status to DATABASE (persists across refreshes)"""
    try:
        db = get_database_connection()
        success = db.update_alert_status(alert_id, status, analyst, notes)
        if success:
           
            st.session_state.alert_statuses[alert_id] = {
                'status': status, 'analyst': analyst,
                'notes': notes, 'updated_at': datetime.now()
            }
        else:
            st.warning(f"⚠️ No alert found with alert_id={alert_id}")
    except Exception as e:
        st.error(f"❌ Failed to update database: {e}")
        # Fallback to session state only
        st.session_state.alert_statuses[alert_id] = {
            'status': status, 'analyst': analyst,
            'notes': notes, 'updated_at': datetime.now()
        }


def render_empty_state(icon="🛡️", title="System Secure",
                       subtitle="No alerts detected in the current time window."):
    st.markdown(f"""
    <div class="empty-state">
        <div class="es-icon">{icon}</div>
        <div class="es-title">{title}</div>
        <div class="es-sub">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)


def render_select_placeholder():
    st.markdown("""
    <div class="select-placeholder">
        <div class="sp-icon">🔍</div>
        <div class="sp-title">Select an alert to begin investigation</div>
        <div class="sp-sub">Click any alert from the queue on the left to view
        full details, features, and related context.</div>
    </div>
    """, unsafe_allow_html=True)


def plotly_layout_defaults(height=350, **extra):
    base = dict(
        height=height,
        margin=dict(t=20, b=20, l=20, r=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#94a3b8', size=11),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(30,41,59,0.8)', zeroline=False),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=10)),
    )
    base.update(extra)
    return base


with st.sidebar:
   
    st.markdown("""
    <div style="text-align:center; padding: 8px 0 12px 0;">
        <div style="font-size: 2.2rem;
                    filter: drop-shadow(0 0 12px rgba(99,102,241,0.4));">🛡️</div>
        <div style="font-size: 1rem; font-weight: 800;
                    letter-spacing: 2px; color: #e2e8f0;">IDS DASHBOARD</div>
        <div style="font-size: 0.65rem; color: #475569;
                    letter-spacing: 2px; margin-top: 2px;">POSTGRESQL · REDIS · v4.1</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

  
    page_index = (PAGES.index(st.session_state.current_page)
                  if st.session_state.current_page in PAGES else 0)
    page = st.selectbox(
        "Navigate to", options=PAGES, index=page_index,
        format_func=lambda p: f"{PAGE_ICONS.get(p, '')}  {p}",
        label_visibility="collapsed"
    )
    st.session_state.current_page = page

    st.markdown("---")

   
    st.markdown("##### 🔍 Filters")
    severity_filter = st.multiselect("Severity", options=SEVERITY_ORDER,
                                     default=SEVERITY_ORDER)
    min_confidence = st.slider("Min Confidence", 0.0, 1.0, 0.0, 0.05,
                               help="Layer 2 Random Forest confidence threshold")
    time_range = st.selectbox(
        "Time Range",
        ['Last 5 min', 'Last 15 min', 'Last 1 hour', 'Last 24 hours', 'All time'],
        index=2
    )
    load_limit = st.slider("Max Alerts", 100, 10000, 1000, 100)
    show_ambiguous = st.checkbox("Include Ambiguous", value=True)

    st.markdown("---")

    
    with st.expander("⚙️ System Settings", expanded=False):
        st.markdown("**Connections**")
        try:
            db = get_database_connection()
            _stats = db.get_alert_statistics()
            st.success("PostgreSQL connected", icon="🟢")
            cache_stats = db.get_cache_stats()
            if cache_stats['total'] > 0:
                st.success(f"Redis ✓  ({cache_stats['hit_rate']:.0f}% hit rate)",
                           icon="🟢")
            else:
                st.info("Redis cache warming up…", icon="🔵")
        except Exception as e:
            st.error("Database disconnected", icon="🔴")
            st.caption(str(e))

        st.markdown("**🔄 Auto-Refresh**")
        
        st.session_state.auto_refresh = st.checkbox(
            "Enable auto-refresh",
            value=st.session_state.auto_refresh,
            key='_cb_autorefresh'
        )
        if st.session_state.auto_refresh:
            st.session_state.refresh_interval = st.slider(
                "Refresh interval (seconds)", 1, 30,
                value=st.session_state.refresh_interval,
                key='_sl_interval'
            )


auto_refresh = st.session_state.auto_refresh
refresh_interval = st.session_state.refresh_interval

df = load_alerts_from_db(
    limit=load_limit,
    severity_filter=severity_filter if severity_filter else None,
    min_confidence=min_confidence
)
layer1_stats = load_layer1_stats()
baselines = load_device_baselines()

if not df.empty:
    if time_range != 'All time' and 'timestamp' in df.columns:
        time_map = {
            'Last 5 min': 5, 'Last 15 min': 15,
            'Last 1 hour': 60, 'Last 24 hours': 1440
        }
        cutoff = datetime.now() - timedelta(minutes=time_map[time_range])
        df = df[df['timestamp'] >= cutoff]
    if not show_ambiguous and 'attack_id' in df.columns:
        df = df[df['attack_id'] != 100]


if auto_refresh:
    badge_html = '<span class="ph-status ph-live">● LIVE</span>'
else:
    badge_html = '<span class="ph-status ph-paused">⏸ PAUSED</span>'

st.markdown(f"""
<div class="page-header">
    <div class="ph-left">
        <div class="ph-icon">{PAGE_ICONS.get(page, '')}</div>
        <div>
            <div class="ph-title">{page}</div>
            <div class="ph-sub">{PAGE_SUBTITLES.get(page, '')}</div>
        </div>
    </div>
    <div class="ph-right">
        {badge_html}
        <div class="ph-time">{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
    </div>
</div>
""", unsafe_allow_html=True)


if page == "Dashboard":

    if df.empty:
        render_empty_state(
            icon="🛡️", title="System Secure",
            subtitle="No alerts detected. The network is operating within "
                     "normal parameters. Adjust filters or run the IDS "
                     "simulation to generate data."
        )
    else:
        total = len(df)
        critical = (len(df[df['severity'] == 'CRITICAL'])
                    if 'severity' in df.columns else 0)
        high = (len(df[df['severity'] == 'HIGH'])
                if 'severity' in df.columns else 0)
        medium = (len(df[df['severity'] == 'MEDIUM'])
                  if 'severity' in df.columns else 0)
        low = (len(df[df['severity'] == 'LOW'])
               if 'severity' in df.columns else 0)
        ambiguous = (len(df[df['attack_id'] == 100])
                     if 'attack_id' in df.columns else 0)
        avg_conf = (df['confidence'].mean()
                    if 'confidence' in df.columns else 0)

        def pct(n):
            return f"{n/total*100:.1f}% of total" if total else "—"

        st.markdown(f"""
        <div class="metric-strip">
            <div class="mc mc-total">
                <div class="mc-icon">📋</div>
                <div class="mc-val">{total:,}</div>
                <div class="mc-label">Total Alerts</div>
            </div>
            <div class="mc mc-crit">
                <div class="mc-icon">🔴</div>
                <div class="mc-val">{critical:,}</div>
                <div class="mc-label">Critical</div>
                <div class="mc-pct">{pct(critical)}</div>
            </div>
            <div class="mc mc-high">
                <div class="mc-icon">🟠</div>
                <div class="mc-val">{high:,}</div>
                <div class="mc-label">High</div>
                <div class="mc-pct">{pct(high)}</div>
            </div>
            <div class="mc mc-med">
                <div class="mc-icon">🟡</div>
                <div class="mc-val">{medium:,}</div>
                <div class="mc-label">Medium</div>
                <div class="mc-pct">{pct(medium)}</div>
            </div>
            <div class="mc mc-low">
                <div class="mc-icon">🔵</div>
                <div class="mc-val">{low:,}</div>
                <div class="mc-label">Low</div>
                <div class="mc-pct">{pct(low)}</div>
            </div>
            <div class="mc mc-amb">
                <div class="mc-icon">❓</div>
                <div class="mc-val">{ambiguous:,}</div>
                <div class="mc-label">Ambiguous</div>
                <div class="mc-pct">{pct(ambiguous)}</div>
            </div>
            <div class="mc mc-conf">
                <div class="mc-icon">🎯</div>
                <div class="mc-val">{avg_conf:.0%}</div>
                <div class="mc-label">Avg Confidence</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        chart_l, chart_r = st.columns(2, gap="large")

        with chart_l:
            st.markdown(
                '<div class="sec-header"><h3>🎯 Attack Distribution</h3></div>',
                unsafe_allow_html=True)
            if 'attack_type' in df.columns:
                ac = df['attack_type'].value_counts()
                fig = px.pie(
                    values=ac.values, names=ac.index,
                    color=ac.index, color_discrete_map=ATTACK_COLORS, hole=0.55)
                fig.update_traces(
                    textposition='inside', textinfo='percent+label',
                    textfont_size=10,
                    hovertemplate='<b>%{label}</b><br>Count: %{value}'
                                  '<br>%{percent}<extra></extra>')
                fig.update_layout(
                    **plotly_layout_defaults(height=360, showlegend=False))
                st.plotly_chart(fig, use_container_width=True)

        with chart_r:
            st.markdown(
                '<div class="sec-header"><h3>⚠️ Severity Breakdown</h3></div>',
                unsafe_allow_html=True)
            if 'severity' in df.columns:
                sc = df['severity'].value_counts().reindex(
                    SEVERITY_ORDER, fill_value=0)
                fig = go.Figure(go.Bar(
                    x=sc.index, y=sc.values,
                    marker_color=[SEVERITY_COLORS.get(s, '#666')
                                  for s in sc.index],
                    text=[f"{v:,}" for v in sc.values],
                    textposition='outside',
                    hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
                ))
                fig.update_layout(
                    **plotly_layout_defaults(height=360, showlegend=False))
                st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            '<div class="sec-header"><h3>📈 Alert Timeline</h3>'
            '<span class="sec-count">alerts per minute</span></div>',
            unsafe_allow_html=True)
        if 'timestamp' in df.columns and len(df) > 0:
            df_time = (df.set_index('timestamp')
                       .resample('1min').size().reset_index())
            df_time.columns = ['timestamp', 'count']
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_time['timestamp'], y=df_time['count'],
                mode='lines', fill='tozeroy',
                line=dict(color='#818cf8', width=2.5, shape='spline'),
                fillcolor='rgba(129,140,248,0.08)',
                hovertemplate='<b>%{x}</b><br>Alerts: %{y}<extra></extra>'
            ))
            fig.update_layout(**plotly_layout_defaults(
                height=220,
                yaxis=dict(showgrid=True, gridcolor='rgba(30,41,59,0.8)',
                           title='Alerts / min')
            ))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            '<div class="sec-header"><h3>🚨 Priority Alerts</h3>'
            '<span class="sec-count">requires attention</span></div>',
            unsafe_allow_html=True)
        if 'severity' in df.columns:
            crit_df = (df[df['severity'].isin(['CRITICAL', 'HIGH'])]
                       .sort_values('timestamp', ascending=False).head(6))
            if not crit_df.empty:
                cols = st.columns(2, gap="medium")
                for idx, (_, alert) in enumerate(crit_df.iterrows()):
                    with cols[idx % 2]:
                        sev = alert.get('severity', 'UNKNOWN')
                        atype = alert.get('attack_type', 'Unknown')
                        conf = alert.get('confidence', 0)
                        ts = (alert['timestamp'].strftime('%H:%M:%S')
                              if 'timestamp' in alert else '—')
                        pid = (int(alert['alert_id'])
                               if 'alert_id' in alert else '—')
                        c_color = confidence_color(conf)
                        st.markdown(f"""
                        <div class="inv-card">
                            <div class="inv-indicator ind-{sev.lower()}"></div>
                            <div class="inv-body">
                                <div class="inv-title">{atype}</div>
                                <div class="inv-sub">
                                    Alert #{pid} ·
                                    <span class="sev-badge {severity_css(sev)}">
                                        {sev}</span>
                                </div>
                            </div>
                            <div class="inv-right">
                                <div class="inv-time">{ts}</div>
                                <div class="inv-conf"
                                     style="color:{c_color}">{conf:.0%}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.success(
                    "✅ No critical or high severity alerts in the "
                    "current time range.")


elif page == "Alerts":

    if df.empty:
        render_empty_state(
            icon="✅", title="No Alerts to Investigate",
            subtitle="All clear! Adjust your filters or time range, "
                     "or run the IDS simulation to populate alerts."
        )
    else:
        tab_all, tab_ambiguous = st.tabs(
            ["🚨 All Alerts", "🔵 Ambiguous Queue"])

        with tab_all:
            # Filter bar
            fb1, fb2, fb3 = st.columns([2.5, 2.5, 2])
            with fb1:
                sort_by = st.selectbox("Sort", [
                    'Time (Newest)', 'Time (Oldest)',
                    'Severity (Critical First)',
                    'Confidence (Highest)', 'Attack Type (A→Z)'
                ], key='sort_all')
            with fb2:
                atk_options = (sorted(df['attack_type'].unique())
                               if 'attack_type' in df.columns else [])
                attack_filter = st.multiselect(
                    "Attack Type", options=atk_options,
                    default=[], key='atk_all')
            with fb3:
                status_filter = st.multiselect(
                    "Status", options=ALERT_STATUS_OPTIONS,
                    default=[], key='status_all')

            display_df = df.copy()
            if (sort_by == 'Time (Newest)'
                    and 'timestamp' in display_df.columns):
                display_df = display_df.sort_values(
                    'timestamp', ascending=False)
            elif (sort_by == 'Time (Oldest)'
                  and 'timestamp' in display_df.columns):
                display_df = display_df.sort_values(
                    'timestamp', ascending=True)
            elif (sort_by == 'Severity (Critical First)'
                  and 'severity' in display_df.columns):
                sev_map = {s: i for i, s in enumerate(SEVERITY_ORDER)}
                display_df['_sev_rank'] = display_df['severity'].map(sev_map)
                display_df = display_df.sort_values(
                    '_sev_rank').drop('_sev_rank', axis=1)
            elif (sort_by == 'Confidence (Highest)'
                  and 'confidence' in display_df.columns):
                display_df = display_df.sort_values(
                    'confidence', ascending=False)
            elif (sort_by == 'Attack Type (A→Z)'
                  and 'attack_type' in display_df.columns):
                display_df = display_df.sort_values('attack_type')

      
            if attack_filter and 'attack_type' in display_df.columns:
                display_df = display_df[
                    display_df['attack_type'].isin(attack_filter)]

        
            if status_filter and 'alert_id' in display_df.columns:
                display_df = display_df[display_df['alert_id'].apply(
                    lambda aid: (get_alert_status(aid)['status']
                                 in status_filter)
                )]

     
            filtered_count = len(display_df)
            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:12px;
                        margin: 8px 0 12px 0;">
                <span style="font-size:0.78rem; color:#64748b;">
                    Showing</span>
                <span style="font-size:1.3rem; font-weight:800;
                             color:#a78bfa;">{filtered_count:,}</span>
                <span style="font-size:0.78rem; color:#64748b;">
                    alerts after filters
                    (from {len(df):,} total)</span>
            </div>
            """, unsafe_allow_html=True)

     
            list_col, detail_col = st.columns([1, 2], gap="large")

         
            with list_col:
                st.markdown(f"""
                <div class="sec-header">
                    <h3>Alert Queue</h3>
                    <span class="sec-count">{filtered_count}</span>
                </div>
                """, unsafe_allow_html=True)

                page_size = 12
                total_pages = max(
                    1, (filtered_count + page_size - 1) // page_size)
                if total_pages > 1:
                    alert_page = st.number_input(
                        "Page", 1, total_pages, 1,
                        key='page_all', label_visibility="collapsed")
                else:
                    alert_page = 1
                start_idx = (alert_page - 1) * page_size
                page_df = display_df.iloc[start_idx:start_idx + page_size]

                if total_pages > 1:
                    st.caption(f"Page {alert_page} of {total_pages}")

                for _, alert in page_df.iterrows():
                    sev = alert.get('severity', 'UNKNOWN')
                    atype = alert.get('attack_type', 'Unknown')
                    conf = alert.get('confidence', 0)
                    ts = (alert['timestamp'].strftime('%H:%M:%S')
                          if 'timestamp' in alert else '—')
                    pid = (int(alert['alert_id'])
                           if 'alert_id' in alert else None)
                    a_status = get_alert_status(pid)

                    si = ""
                    if a_status['status'] == 'RESOLVED':
                        si = " ✅"
                    elif a_status['status'] == 'INVESTIGATING':
                        si = " 🔍"
                    elif a_status['status'] == 'REVIEWED':
                        si = " 👁️"

                    btn_label = (f"{severity_emoji(sev)} {atype}  ·  "
                                 f"{ts}  ·  {conf:.0%}{si}")
                    if pid is not None and st.button(
                            btn_label, key=f"sel_{pid}",
                            use_container_width=True):
                        st.session_state.selected_alert_id = pid

            with detail_col:
                sel_id = st.session_state.selected_alert_id

                if (sel_id is not None
                        and 'alert_id' in display_df.columns
                        and sel_id in display_df['alert_id'].values):
                    alert = display_df[
                        display_df['alert_id'] == sel_id].iloc[0]
                    sev = alert.get('severity', 'UNKNOWN')
                    atype = alert.get('attack_type', 'Unknown')
                    conf = alert.get('confidence', 0)
                    ts = (alert['timestamp'].strftime(
                              '%Y-%m-%d %H:%M:%S')
                          if 'timestamp' in alert else '—')
                    pid = int(alert['alert_id'])
                    aid = alert.get('attack_id', '—')
                    desc = alert.get('description', '')
                    current_status = get_alert_status(pid)
                    c_color = confidence_color(conf)

                    # Detail panel
                    st.markdown(f"""
                    <div class="detail-panel">
                        <div class="dp-header">
                            <div>
                                <div class="dp-title">{atype}</div>
                                <div style="margin-top:8px;">
                                    <span class="sev-badge
                                        {severity_css(sev)}">{sev}</span>
                                    <span class="stat-badge
                                        {status_css(current_status['status'])}"
                                        style="margin-left:6px;">
                                        {current_status['status']}</span>
                                </div>
                            </div>
                            <div class="dp-id">Alert #{pid}</div>
                        </div>
                        <div class="dp-grid">
                            <div class="dp-field">
                                <div class="dp-flabel">Timestamp</div>
                                <div class="dp-fvalue">{ts}</div>
                            </div>
                            <div class="dp-field">
                                <div class="dp-flabel">Attack ID</div>
                                <div class="dp-fvalue">{aid}</div>
                            </div>
                            <div class="dp-field">
                                <div class="dp-flabel">Confidence</div>
                                <div class="dp-fvalue"
                                     style="color:{c_color}">
                                    {conf:.2%}</div>
                            </div>
                            <div class="dp-field">
                                <div class="dp-flabel">Severity</div>
                                <div class="dp-fvalue">
                                    {severity_emoji(sev)} {sev}</div>
                            </div>
                        </div>
                        <div style="padding:0 2px;">
                            <div style="display:flex;
                                        justify-content:space-between;
                                        font-size:0.75rem;
                                        color:#64748b;
                                        margin-bottom:2px;">
                                <span>Confidence</span>
                                <span style="color:{c_color};
                                             font-weight:600;">
                                    {conf:.1%}</span>
                            </div>
                            <div class="conf-bar-outer">
                                <div class="conf-bar-inner"
                                     style="width:{conf*100}%;
                                            background:{c_color};"></div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    if desc:
                        st.markdown("##### 📝 Alert Explanation")
                        st.info(desc)

                    features = alert.get('features', None)
                    if features and isinstance(features, dict):
                        st.markdown("##### 🔬 Key Features")
                        priority_keys = [
                            'Rate', 'Protocol Type',
                            'Time_To_Live', 'Packet_Length']
                        display_keys = (
                            [k for k in priority_keys if k in features]
                            + [k for k in features
                               if k not in priority_keys])
                        feat_cols = st.columns(
                            min(4, max(1, len(display_keys[:4]))))
                        for i, key in enumerate(display_keys[:4]):
                            with feat_cols[i % len(feat_cols)]:
                                val = features[key]
                                dval = (f"{val:.2f}"
                                        if isinstance(val, (int, float))
                                        else str(val))
                                st.metric(key, dval)

                        with st.expander("📋 View All Features",
                                         expanded=False):
                            feat_df = pd.DataFrame(
                                list(features.items()),
                                columns=['Feature', 'Value'])
                            st.dataframe(feat_df,
                                         use_container_width=True,
                                         height=250, hide_index=True)

                    st.markdown("---")

         
                    st.markdown("##### ✅ Investigation Actions")
                    act1, act2 = st.columns(2)
                    with act1:
                        cur_idx = (
                            ALERT_STATUS_OPTIONS.index(
                                current_status['status'])
                            if current_status['status']
                               in ALERT_STATUS_OPTIONS
                            else 0)
                        new_status = st.selectbox(
                            "Update Status",
                            options=ALERT_STATUS_OPTIONS,
                            index=cur_idx, key=f"ack_st_{pid}")
                    with act2:
                        analyst_name = st.text_input(
                            "Analyst Name",
                            value=current_status.get('analyst', ''),
                            key=f"ack_an_{pid}")

                    notes = st.text_area(
                        "Investigation Notes",
                        value=current_status.get('notes', ''),
                        placeholder="Document your findings, actions "
                                    "taken, recommended follow-ups…",
                        height=80, key=f"ack_nt_{pid}")

                    btn1, btn2, btn3 = st.columns(3)
                    with btn1:
                        if st.button("💾 Save Status", type="primary",
                                     use_container_width=True,
                                     key=f"ack_save_{pid}"):
                            set_alert_status(
                                pid, new_status, analyst_name, notes)
                            st.success(
                                f"Alert #{pid} → **{new_status}**")
                    with btn2:
                        if st.button("✅ Quick Resolve",
                                     use_container_width=True,
                                     key=f"ack_res_{pid}"):
                            set_alert_status(
                                pid, 'RESOLVED', analyst_name, notes)
                            st.success(
                                f"Alert #{pid} → **RESOLVED**")
                    with btn3:
                        if st.button("🔍 Mark Investigating",
                                     use_container_width=True,
                                     key=f"ack_inv_{pid}"):
                            set_alert_status(
                                pid, 'INVESTIGATING',
                                analyst_name, notes)
                            st.info(
                                f"Alert #{pid} → **INVESTIGATING**")
                else:
                    render_select_placeholder()

            st.markdown("---")
            exp1, exp2 = st.columns(2)
            with exp1:
                dcols = [c for c in [
                    'timestamp', 'severity', 'attack_type',
                    'confidence', 'attack_id', 'alert_id'
                ] if c in display_df.columns]
                if dcols:
                    csv_data = display_df[dcols].to_csv(index=False)
                    st.download_button(
                        "📥 Export CSV", csv_data,
                        f"ids_alerts_"
                        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv", use_container_width=True)
            with exp2:
                if st.button("📊 View Statistics Summary",
                             use_container_width=True,
                             key='stats_all'):
                    stat_data = {
                        'Total Alerts': len(display_df),
                        'Critical': (
                            len(display_df[
                                display_df['severity'] == 'CRITICAL'])
                            if 'severity' in display_df.columns
                            else 0),
                        'High': (
                            len(display_df[
                                display_df['severity'] == 'HIGH'])
                            if 'severity' in display_df.columns
                            else 0),
                        'Medium': (
                            len(display_df[
                                display_df['severity'] == 'MEDIUM'])
                            if 'severity' in display_df.columns
                            else 0),
                        'Low': (
                            len(display_df[
                                display_df['severity'] == 'LOW'])
                            if 'severity' in display_df.columns
                            else 0),
                        'Avg Confidence': (
                            f"{display_df['confidence'].mean():.2%}"
                            if 'confidence' in display_df.columns
                            else 'N/A'),
                    }
                    st.json(stat_data)


        with tab_ambiguous:
            ambiguous_df = (
                df[df['attack_id'] == 100]
                if 'attack_id' in df.columns
                else pd.DataFrame())

            aq1, aq2, aq3, aq4 = st.columns(4)
            with aq1:
                st.metric("🔵 Total Ambiguous", len(ambiguous_df))
            with aq2:
                new_ct = (
                    sum(1 for _, a in ambiguous_df.iterrows()
                        if 'alert_id' in a
                        and get_alert_status(
                            int(a['alert_id']))['status'] == 'NEW')
                    if not ambiguous_df.empty else 0)
                st.metric("🆕 Unreviewed", new_ct)
            with aq3:
                inv_ct = (
                    sum(1 for _, a in ambiguous_df.iterrows()
                        if 'alert_id' in a
                        and get_alert_status(
                            int(a['alert_id']))['status'] == 'INVESTIGATING')
                    if not ambiguous_df.empty else 0)
                st.metric("🔍 Investigating", inv_ct)
            with aq4:
                res_ct = (
                    sum(1 for _, a in ambiguous_df.iterrows()
                        if 'alert_id' in a
                        and get_alert_status(
                            int(a['alert_id']))['status']
                        in ['RESOLVED', 'FALSE POSITIVE'])
                    if not ambiguous_df.empty else 0)
                st.metric("✅ Resolved / FP", res_ct)

            st.markdown("---")

            if ambiguous_df.empty:
                render_empty_state(
                    icon="✅", title="No Ambiguous Alerts",
                    subtitle="All threats have been classified by "
                             "the ML pipeline. No unknown patterns "
                             "detected.")
            else:
                sort_amb = st.radio(
                    "Sort by",
                    ['Confidence (High → Low)',
                     'Time (Newest)', 'Time (Oldest)'],
                    horizontal=True, key='sort_ambig')
                if (sort_amb == 'Confidence (High → Low)'
                        and 'confidence' in ambiguous_df.columns):
                    ambiguous_df = ambiguous_df.sort_values(
                        'confidence', ascending=False)
                elif (sort_amb == 'Time (Newest)'
                      and 'timestamp' in ambiguous_df.columns):
                    ambiguous_df = ambiguous_df.sort_values(
                        'timestamp', ascending=False)
                elif (sort_amb == 'Time (Oldest)'
                      and 'timestamp' in ambiguous_df.columns):
                    ambiguous_df = ambiguous_df.sort_values(
                        'timestamp', ascending=True)

                st.markdown(
                    "<div style='height:8px'></div>",
                    unsafe_allow_html=True)

                for _, alert in ambiguous_df.head(25).iterrows():
                    conf = alert.get('confidence', 0)
                    ts = (alert['timestamp'].strftime(
                              '%Y-%m-%d %H:%M:%S')
                          if 'timestamp' in alert else '—')
                    pid = (int(alert['alert_id'])
                           if 'alert_id' in alert else None)
                    a_status = get_alert_status(pid)
                    c_color = confidence_color(conf)

                    status_icon = {
                        'NEW': '🆕', 'REVIEWED': '👁️',
                        'INVESTIGATING': '🔍',
                        'RESOLVED': '✅',
                        'FALSE POSITIVE': '🚫'
                    }.get(a_status['status'], '❓')

                    with st.expander(
                        f"{status_icon}  Alert #{pid}  ·  "
                        f"Confidence: {conf:.1%}  ·  {ts}  ·  "
                        f"{a_status['status']}",
                        expanded=False
                    ):
                        inv_left, inv_right = st.columns(
                            [3, 2], gap="large")

                        with inv_left:
                            st.markdown("**🔍 Threat Assessment**")
                            st.markdown(f"""
                            <div style="background:#0f172a;
                                        border:1px solid #1e293b;
                                        border-radius:10px;
                                        padding:16px;
                                        margin-bottom:12px;">
                                <div style="display:flex; gap:12px;
                                            align-items:center;
                                            margin-bottom:8px;">
                                    <span class="sev-badge sev-medium">
                                        AMBIGUOUS</span>
                                    <span style="font-size:0.8rem;
                                                 color:#64748b;">
                                        Attack ID: 100</span>
                                </div>
                                <ul style="margin:0;
                                           padding-left:18px;
                                           color:#94a3b8;
                                           font-size:0.85rem;
                                           line-height:1.8;">
                                    <li>⚠️ Unknown / unclassified
                                        attack pattern</li>
                                    <li>🧬 Potential zero-day exploit
                                        signature</li>
                                    <li>🔬 Novel behavior — requires
                                        manual investigation</li>
                                </ul>
                            </div>
                            """, unsafe_allow_html=True)

                            st.markdown(f"""
                            <div style="margin-bottom:16px;">
                                <div style="display:flex;
                                            justify-content:space-between;
                                            font-size:0.78rem;
                                            color:#64748b;">
                                    <span>Model Confidence</span>
                                    <span style="color:{c_color};
                                                 font-weight:700;">
                                        {conf:.2%}</span>
                                </div>
                                <div class="conf-bar-outer">
                                    <div class="conf-bar-inner"
                                         style="width:{conf*100}%;
                                                background:{c_color};">
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                            features = alert.get('features', None)
                            if features and isinstance(features, dict):
                                st.markdown("**🧬 Key Features**")
                                priority_keys = [
                                    'Rate', 'Protocol Type',
                                    'Time_To_Live', 'Packet_Length']
                                show_keys = (
                                    [k for k in priority_keys
                                     if k in features]
                                    + [k for k in features
                                       if k not in priority_keys])
                                feat_cols = st.columns(
                                    min(3, max(1, len(show_keys[:3]))))
                                for i, key in enumerate(show_keys[:3]):
                                    with feat_cols[
                                            i % len(feat_cols)]:
                                        val = features[key]
                                        dval = (
                                            f"{val:.2f}"
                                            if isinstance(
                                                val, (int, float))
                                            else str(val))
                                        st.metric(key, dval)

                                if st.checkbox(
                                        "Show all features",
                                        key=f"amb_feat_{pid}"):
                                    feat_df = pd.DataFrame(
                                        list(features.items()),
                                        columns=['Feature', 'Value'])
                                    st.dataframe(
                                        feat_df,
                                        use_container_width=True,
                                        hide_index=True)

                        with inv_right:
                            st.markdown("**✏️ Investigation Actions**")
                            amb_idx = (
                                AMBIGUOUS_STATUS_OPTIONS.index(
                                    a_status['status'])
                                if a_status['status']
                                   in AMBIGUOUS_STATUS_OPTIONS
                                else 0)
                            new_st = st.selectbox(
                                "Status",
                                AMBIGUOUS_STATUS_OPTIONS,
                                index=amb_idx,
                                key=f"amb_st_{pid}")
                            analyst = st.text_input(
                                "Analyst",
                                value=a_status.get('analyst', ''),
                                key=f"amb_an_{pid}")
                            note = st.text_area(
                                "Notes",
                                value=a_status.get('notes', ''),
                                placeholder="Document investigation "
                                            "findings…",
                                height=80, key=f"amb_nt_{pid}")

                            bc1, bc2 = st.columns(2)
                            with bc1:
                                if st.button(
                                        "💾 Save",
                                        key=f"amb_save_{pid}",
                                        use_container_width=True,
                                        type="primary"):
                                    set_alert_status(
                                        pid, new_st, analyst, note)
                                    st.success(f"#{pid} → {new_st}")
                            with bc2:
                                if st.button(
                                        "🚫 False Positive",
                                        key=f"amb_fp_{pid}",
                                        use_container_width=True):
                                    set_alert_status(
                                        pid, 'FALSE POSITIVE',
                                        analyst, note)
                                    st.success(
                                        f"#{pid} → FALSE POSITIVE")


elif page == "ML Analytics":

    st.markdown(
        '<div class="sec-header">'
        '<h3>🧠 Layer 1 — Unsupervised Anomaly Detection</h3>'
        '</div>',
        unsafe_allow_html=True)

    if layer1_stats:
        total = layer1_stats.get('total_alerts', 0)

        ml1, ml2, ml3, ml4 = st.columns(4, gap="medium")
        ml_items = [
            (ml1, 'ocsvm_anomalies', 'OCSVM Anomalies', '🔮'),
            (ml2, 'lof_anomalies', 'LOF Anomalies', '📡'),
            (ml3, 'if_anomalies', 'Isolation Forest', '🌲'),
        ]
        for col_widget, key, label, icon in ml_items:
            with col_widget:
                v = layer1_stats.get(key, 0)
                pct_val = v / total * 100 if total else 0
                st.markdown(f"""
                <div class="ml-card">
                    <div class="ml-icon">{icon}</div>
                    <div class="ml-val">{v:,}</div>
                    <div class="ml-label">{label}</div>
                    <div class="ml-pct">{pct_val:.1f}% flagged</div>
                </div>
                """, unsafe_allow_html=True)
        with ml4:
            avg = layer1_stats.get('avg_ensemble_score', 0)
            st.markdown(f"""
            <div class="ml-card">
                <div class="ml-icon">🎯</div>
                <div class="ml-val">{avg:.3f}</div>
                <div class="ml-label">Avg Ensemble Score</div>
                <div class="ml-pct">Combined model output</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(
            "<div style='height:20px'></div>",
            unsafe_allow_html=True)

        st.markdown(
            '<div class="sec-header">'
            '<h3>🗳️ Voting Consensus</h3>'
            '<span class="sec-count">algorithm agreement</span>'
            '</div>',
            unsafe_allow_html=True)

        vote_labels = [
            '0 Votes (All Benign)', '1 Vote (Likely Benign)',
            '2 Votes (Suspicious)', '3 Votes (Anomaly)']
        vote_values = [
            layer1_stats.get(f'vote_{i}', 0) for i in range(4)]
        vote_colors = ['#10b981', '#3b82f6', '#eab308', '#ef4444']

        fig_v = go.Figure(go.Bar(
            x=vote_labels, y=vote_values,
            marker_color=vote_colors,
            text=[f"{v:,}" for v in vote_values],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Count: %{y:,}'
                          '<extra></extra>'
        ))
        fig_v.update_layout(
            **plotly_layout_defaults(height=320, showlegend=False))
        st.plotly_chart(fig_v, use_container_width=True)

        st.markdown(
            '<div class="sec-header">'
            '<h3>📊 Algorithm Detection Rates</h3></div>',
            unsafe_allow_html=True)
        algo_names = ['OCSVM', 'LOF', 'Isolation Forest']
        algo_vals = [
            (layer1_stats.get('ocsvm_anomalies', 0)
             / total * 100 if total else 0),
            (layer1_stats.get('lof_anomalies', 0)
             / total * 100 if total else 0),
            (layer1_stats.get('if_anomalies', 0)
             / total * 100 if total else 0),
        ]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=algo_vals + [algo_vals[0]],
            theta=algo_names + [algo_names[0]],
            fill='toself',
            fillcolor='rgba(129,140,248,0.15)',
            line=dict(color='#818cf8', width=2),
            name='Detection Rate (%)'
        ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor='rgba(0,0,0,0)',
                radialaxis=dict(
                    visible=True, gridcolor='#1e293b',
                    linecolor='#1e293b'),
                angularaxis=dict(
                    gridcolor='#1e293b', linecolor='#1e293b')
            ),
            **plotly_layout_defaults(height=350, showlegend=True)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    else:
        render_empty_state(
            icon="🧠", title="No Layer 1 Statistics",
            subtitle="Unsupervised anomaly detection data is not yet "
                     "available. Ensure the ML pipeline has processed "
                     "data.")

    st.markdown("---")

    st.markdown(
        '<div class="sec-header">'
        '<h3>🎯 Layer 2 — Supervised Classification '
        '(Random Forest)</h3></div>',
        unsafe_allow_html=True)

    if not df.empty:
        l2_left, l2_right = st.columns(2, gap="large")

        with l2_left:
            st.markdown("**Confidence Distribution**")
            if 'confidence' in df.columns:
                fig = px.histogram(
                    df, x='confidence', nbins=40,
                    color_discrete_sequence=['#818cf8'],
                    marginal='rug')
                fig.update_layout(
                    **plotly_layout_defaults(height=320))
                st.plotly_chart(fig, use_container_width=True)

        with l2_right:
            st.markdown("**Confidence Statistics**")
            if 'confidence' in df.columns:
                s1, s2 = st.columns(2)
                with s1:
                    st.metric("Maximum",
                              f"{df['confidence'].max():.2%}")
                    st.metric("Mean",
                              f"{df['confidence'].mean():.2%}")
                    st.metric("Non-null",
                              f"{df['confidence'].notna().sum():,}")
                with s2:
                    st.metric("Minimum",
                              f"{df['confidence'].min():.2%}")
                    st.metric("Std Dev",
                              f"{df['confidence'].std():.2%}")
                    st.metric("Median",
                              f"{df['confidence'].median():.2%}")

        if ('confidence' in df.columns
                and 'attack_type' in df.columns):
            st.markdown(
                '<div class="sec-header">'
                '<h3>🎯 Confidence by Attack Type</h3></div>',
                unsafe_allow_html=True)
            fig = px.box(
                df, x='attack_type', y='confidence',
                color='attack_type',
                color_discrete_map=ATTACK_COLORS,
                points='outliers')
            fig.update_layout(**plotly_layout_defaults(
                height=380, showlegend=False,
                xaxis=dict(showgrid=False, title=''),
                yaxis=dict(showgrid=True,
                           gridcolor='rgba(30,41,59,0.8)',
                           title='Confidence')
            ))
            st.plotly_chart(fig, use_container_width=True)
    else:
        render_empty_state(
            icon="🎯", title="No Layer 2 Data",
            subtitle="Supervised classification results are not "
                     "yet available.")


elif page == "Feature Analysis":

    if df.empty:
        render_empty_state(
            icon="🧬", title="No Data Loaded",
            subtitle="Adjust filters or run the IDS simulation to "
                     "generate feature data for analysis.")
    elif ('features' not in df.columns
          or df['features'].isna().all()):
        render_empty_state(
            icon="🧬", title="No Feature Data",
            subtitle="Alert records do not contain feature "
                     "information. Ensure the ML pipeline exports "
                     "features.")
    else:
        all_features = [
            f for f in df['features'].dropna()
            if isinstance(f, dict)]
        if not all_features:
            render_empty_state(
                icon="🧬", title="No Features",
                subtitle="No valid feature dictionaries found.")
        else:
            features_df = pd.DataFrame(all_features)
            numeric_cols = sorted(
                features_df.select_dtypes(
                    include=[np.number]).columns.tolist())
            all_cols = (numeric_cols if numeric_cols
                        else sorted(features_df.columns.tolist()))

            st.markdown(
                '<div class="sec-header">'
                '<h3>📊 Feature Statistics Overview</h3></div>',
                unsafe_allow_html=True)
            st.dataframe(
                features_df.describe().round(4),
                use_container_width=True, height=280)

            st.markdown("---")

            sel_col, viz_col = st.columns([1, 3], gap="large")

            with sel_col:
                st.markdown("**Select Feature**")
                selected_feature = st.selectbox(
                    "Feature", options=all_cols,
                    label_visibility="collapsed")

                if (selected_feature
                        and selected_feature in features_df.columns):
                    col_data = features_df[selected_feature].dropna()
                    is_numeric = col_data.dtype in [
                        np.float64, np.int64, float, int]

                    if is_numeric:
                        st.markdown("**Quick Stats**")
                        st.metric("Mean",
                                  f"{col_data.mean():.4f}")
                        st.metric("Median",
                                  f"{col_data.median():.4f}")
                        st.metric("Std Dev",
                                  f"{col_data.std():.4f}")
                        st.metric("Min / Max",
                                  f"{col_data.min():.2f} — "
                                  f"{col_data.max():.2f}")
                        st.metric("Non-null",
                                  f"{len(col_data):,}")

                        has_bl = (
                            baselines
                            and selected_feature in baselines
                            and isinstance(
                                baselines.get(selected_feature),
                                dict))
                        if has_bl:
                            bl = baselines[selected_feature]
                            bl_mean = bl.get('mean', 0)
                            bl_std = bl.get('std', 1) or 1
                            deviation = abs(
                                col_data.mean() - bl_mean)
                            z_score = deviation / bl_std

                            st.markdown("---")
                            st.markdown("**🔒 Baseline**")
                            st.metric("Baseline Mean",
                                      f"{bl_mean:.4f}")
                            st.metric("Baseline Std",
                                      f"{bl.get('std', 0):.4f}")

                            # Deviation verdict
                            if z_score > 3:
                                verdict_class = "dg-danger"
                                verdict_text = "⚠️ CRITICAL"
                            elif z_score > 2:
                                verdict_class = "dg-warning"
                                verdict_text = "⚡ SUSPICIOUS"
                            else:
                                verdict_class = "dg-normal"
                                verdict_text = "✅ NORMAL"

                            z_color = confidence_color(
                                max(0, 1 - z_score / 5))
                            st.markdown(f"""
                            <div class="dev-gauge">
                                <div class="dg-value"
                                     style="color:{z_color}">
                                    {z_score:.2f}σ</div>
                                <div class="dg-label">
                                    Deviation from Baseline</div>
                                <div class="dg-verdict
                                    {verdict_class}">
                                    {verdict_text}</div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.markdown("**Value Counts**")
                        for val, count in (
                                col_data.value_counts()
                                .head(10).items()):
                            st.caption(f"**{val}**: {count:,}")

            with viz_col:
                if (selected_feature
                        and selected_feature
                        in features_df.columns):
                    col_data = features_df[
                        selected_feature].dropna()
                    is_numeric = col_data.dtype in [
                        np.float64, np.int64, float, int]

                    if is_numeric:
                       
                        st.markdown(
                            f'<div class="sec-header">'
                            f'<h3>📈 Distribution: '
                            f'{selected_feature}</h3>'
                            f'<span class="sec-count">'
                            f'with baseline overlay</span>'
                            f'</div>',
                            unsafe_allow_html=True)

                        has_bl = (
                            baselines
                            and selected_feature in baselines
                            and isinstance(
                                baselines.get(selected_feature),
                                dict))

                      
                        if has_bl:
                            st.markdown("""
                            <div class="baseline-legend">
                                <div class="bl-item">
                                    <span class="bl-dot"
                                          style="background:#818cf8;">
                                    </span> Live Data</div>
                                <div class="bl-item">
                                    <span class="bl-dot"
                                          style="background:#ef4444;">
                                    </span> Baseline Mean</div>
                                <div class="bl-item">
                                    <span class="bl-dot"
                                          style="background:
                                          rgba(239,68,68,0.3);">
                                    </span> Baseline ±1σ</div>
                            </div>
                            """, unsafe_allow_html=True)

                        fig_h = go.Figure()

                     
                        fig_h.add_trace(go.Histogram(
                            x=col_data, nbinsx=40,
                            marker_color='rgba(129,140,248,0.6)',
                            marker_line=dict(
                                color='#818cf8', width=1),
                            name='Live Data',
                            hovertemplate='Value: %{x}<br>'
                                          'Count: %{y}'
                                          '<extra></extra>'
                        ))

                       
                        if has_bl:
                            bl = baselines[selected_feature]
                            bl_mean = bl.get('mean', 0)
                            bl_std = bl.get('std', 0)

                            # Baseline mean line
                            fig_h.add_vline(
                                x=bl_mean, line_width=2.5,
                                line_dash="dash",
                                line_color="#ef4444",
                                annotation_text=(
                                    f"Baseline: {bl_mean:.2f}"),
                                annotation_position="top right",
                                annotation_font_color="#ef4444",
                                annotation_font_size=11)

                        
                            if bl_std > 0:
                                fig_h.add_vrect(
                                    x0=bl_mean - bl_std,
                                    x1=bl_mean + bl_std,
                                    fillcolor=(
                                        "rgba(239,68,68,0.06)"),
                                    line=dict(
                                        color=(
                                            "rgba(239,68,68,0.25)"),
                                        width=1, dash="dot"),
                                    annotation_text="±1σ",
                                    annotation_position="top left",
                                    annotation_font_color="#94a3b8",
                                    annotation_font_size=9)

                        fig_h.update_layout(
                            **plotly_layout_defaults(height=360))
                        st.plotly_chart(
                            fig_h, use_container_width=True)

                        
                        if has_bl:
                            bl = baselines[selected_feature]
                            bl_mean = bl.get('mean', 0)
                            bl_std = bl.get('std', 1) or 1
                            live_mean = col_data.mean()
                            z_score = (abs(live_mean - bl_mean)
                                       / bl_std)

                            
                            st.markdown(
                                '<div class="sec-header">'
                                '<h3>📏 Deviation Gauge</h3>'
                                '<span class="sec-count">'
                                'live mean vs baseline</span>'
                                '</div>',
                                unsafe_allow_html=True)

                            fig_dev = go.Figure()

                           
                            fig_dev.add_trace(go.Bar(
                                x=[5], y=['Deviation'],
                                orientation='h',
                                marker_color='rgba(239,68,68,0.08)',
                                name='Critical (>3σ)',
                                hoverinfo='skip',
                                showlegend=True))
                            fig_dev.add_trace(go.Bar(
                                x=[3], y=['Deviation'],
                                orientation='h',
                                marker_color=(
                                    'rgba(234,179,8,0.10)'),
                                name='Warning (2-3σ)',
                                hoverinfo='skip',
                                showlegend=True))
                            fig_dev.add_trace(go.Bar(
                                x=[2], y=['Deviation'],
                                orientation='h',
                                marker_color=(
                                    'rgba(16,185,129,0.10)'),
                                name='Normal (0-2σ)',
                                hoverinfo='skip',
                                showlegend=True))

                           
                            z_capped = min(z_score, 5)
                            if z_score > 3:
                                marker_color = '#ef4444'
                            elif z_score > 2:
                                marker_color = '#eab308'
                            else:
                                marker_color = '#10b981'

                            fig_dev.add_trace(go.Scatter(
                                x=[z_capped],
                                y=['Deviation'],
                                mode='markers+text',
                                marker=dict(
                                    size=18,
                                    color=marker_color,
                                    symbol='diamond',
                                    line=dict(
                                        color='white', width=2)),
                                text=[f"{z_score:.2f}σ"],
                                textposition='top center',
                                textfont=dict(
                                    size=13, color=marker_color),
                                name=f'Current: {z_score:.2f}σ',
                                hovertemplate=(
                                    f'Deviation: {z_score:.2f}σ'
                                    f'<br>Live Mean: '
                                    f'{live_mean:.4f}'
                                    f'<br>Baseline Mean: '
                                    f'{bl_mean:.4f}'
                                    f'<extra></extra>')
                            ))

                            fig_dev.update_layout(
                                barmode='overlay',
                                height=140,
                                margin=dict(
                                    t=30, b=10, l=10, r=10),
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='#94a3b8'),
                                xaxis=dict(
                                    range=[0, 5.5],
                                    title='Standard Deviations (σ)',
                                    showgrid=True,
                                    gridcolor='rgba(30,41,59,0.5)',
                                    dtick=1),
                                yaxis=dict(showticklabels=False),
                                legend=dict(
                                    orientation='h',
                                    yanchor='bottom',
                                    y=1.02, xanchor='left', x=0,
                                    bgcolor='rgba(0,0,0,0)',
                                    font=dict(size=10))
                            )
                            st.plotly_chart(
                                fig_dev, use_container_width=True)

        
                        if 'attack_type' in df.columns:
                            st.markdown(
                                f'<div class="sec-header">'
                                f'<h3>🎯 {selected_feature} '
                                f'by Attack Type</h3></div>',
                                unsafe_allow_html=True)
                            rows = []
                            for _, row in df.iterrows():
                                feats = row.get('features', None)
                                if (feats
                                        and isinstance(feats, dict)
                                        and selected_feature
                                        in feats):
                                    val = feats[selected_feature]
                                    if isinstance(
                                            val, (int, float)):
                                        rows.append({
                                            'attack_type': row.get(
                                                'attack_type',
                                                'Unknown'),
                                            selected_feature: val
                                        })
                            if rows:
                                fa_df = pd.DataFrame(rows)
                                fig_b = px.box(
                                    fa_df,
                                    x='attack_type',
                                    y=selected_feature,
                                    color='attack_type',
                                    color_discrete_map=ATTACK_COLORS,
                                    points='outliers')

                               
                                has_bl_for_box = (
                                    baselines
                                    and selected_feature in baselines
                                    and isinstance(
                                        baselines.get(
                                            selected_feature), dict))
                                if has_bl_for_box:
                                    bl_m = baselines[
                                        selected_feature].get(
                                            'mean', 0)
                                    fig_b.add_hline(
                                        y=bl_m, line_width=2,
                                        line_dash="dash",
                                        line_color="#ef4444",
                                        annotation_text=(
                                            f"Baseline: "
                                            f"{bl_m:.2f}"),
                                        annotation_position=(
                                            "top right"),
                                        annotation_font_color=(
                                            "#ef4444"),
                                        annotation_font_size=10)

                                fig_b.update_layout(
                                    **plotly_layout_defaults(
                                        height=380,
                                        showlegend=False,
                                        xaxis=dict(
                                            showgrid=False,
                                            title=''),
                                        yaxis=dict(
                                            showgrid=True,
                                            gridcolor=(
                                                'rgba(30,41,59,'
                                                '0.8)'))
                                    ))
                                st.plotly_chart(
                                    fig_b,
                                    use_container_width=True)
                    else:
                       
                        st.markdown(
                            f'<div class="sec-header">'
                            f'<h3>📊 Value Distribution: '
                            f'{selected_feature}</h3></div>',
                            unsafe_allow_html=True)
                        vc = col_data.value_counts().head(15)
                        fig_c = go.Figure(go.Bar(
                            x=vc.index.astype(str),
                            y=vc.values,
                            marker_color=(
                                'rgba(129,140,248,0.7)'),
                            marker_line=dict(
                                color='#818cf8', width=1),
                            text=[f"{v:,}" for v in vc.values],
                            textposition='outside',
                            hovertemplate=(
                                '<b>%{x}</b><br>'
                                'Count: %{y:,}<extra></extra>')
                        ))
                        fig_c.update_layout(
                            **plotly_layout_defaults(height=360))
                        st.plotly_chart(
                            fig_c, use_container_width=True)

            st.markdown("---")

            
            st.markdown(
                '<div class="sec-header">'
                '<h3>🔥 Feature Correlation Heatmap</h3>'
                '<span class="sec-count">'
                'top 15 by variance</span></div>',
                unsafe_allow_html=True)
            numeric_feats = features_df.select_dtypes(
                include=[np.number])
            if len(numeric_feats.columns) >= 2:
                top_cols = (numeric_feats.var()
                            .sort_values(ascending=False)
                            .head(15).index.tolist())
                corr = numeric_feats[top_cols].corr()
                fig_corr = go.Figure(data=go.Heatmap(
                    z=corr.values,
                    x=corr.columns, y=corr.columns,
                    colorscale='RdBu_r', zmid=0,
                    zmin=-1, zmax=1,
                    text=np.round(corr.values, 2),
                    texttemplate='%{text}',
                    textfont=dict(size=9),
                    hovertemplate=(
                        '%{x} vs %{y}<br>'
                        'Correlation: %{z:.3f}<extra></extra>')
                ))
                fig_corr.update_layout(
                    **plotly_layout_defaults(height=520))
                st.plotly_chart(
                    fig_corr, use_container_width=True)
            else:
                st.info("Need at least 2 numeric features for "
                        "correlation analysis.")




alert_count_text = (
    f"Showing {len(df):,} alerts from PostgreSQL"
    if not df.empty else "No alerts loaded")
st.markdown(f"""
<div class="dash-footer">
    <span>Last Updated: {datetime.now().strftime(
        '%Y-%m-%d %H:%M:%S')}</span>
    <span>{alert_count_text}</span>
    <span>🛡️ IDS Dashboard</span>
</div>
""", unsafe_allow_html=True)


if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()