"""
Real-Time IDS Alert Dashboard
Streamlit interface for monitoring JSONL alert stream
Name: Tan Yi Feng
ID: 23WMR14766
"""

import streamlit as st
import pandas as pd
import json
import time
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import numpy as np


st.set_page_config(
    page_title="IDS Alert Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    .critical-alert {
        background-color: #ff4444;
        color: white;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .high-alert {
        background-color: #ff8800;
        color: white;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .medium-alert {
        background-color: #ffbb33;
        color: black;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .low-alert {
        background-color: #00C851;
        color: white;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .ambiguous-alert {
        background-color: #33b5e5;
        color: white;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

ATTACK_COLORS = {
    'ARP Spoofing': '#FFA500',
    'MQTT Connect Flood': '#FF0000',
    'MQTT Publish Flood': '#DC143C',
    'MQTT Malformed': '#FF6347',
    'Reconnaissance': '#FFD700',
    'Recon (VulnScan)': '#DAA520',
    'ICMP Flood': '#8B0000',
    'SYN Flood': '#B22222',
    'TCP Flood': '#CD5C5C',
    'UDP Flood': '#F08080',
    'Ambiguous': '#00BFFF'
}

SEVERITY_COLORS = {
    'CRITICAL': '#FF0000',
    'HIGH': '#FF8800',
    'MEDIUM': '#FFBB33',
    'LOW': '#33B5E5',
    'INFO': '#00C851'
}

SEVERITY_ORDER = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'INFO']


@st.cache_data(ttl=1)
def load_alerts(filepath='alerts.jsonl', max_lines=None):
    """Load alerts from JSONL file"""
    alerts = []
    try:
        with open(filepath, 'r') as f:
            for i, line in enumerate(f):
                if max_lines and i >= max_lines:
                    break
                if line.strip():
                    alert = json.loads(line)
                    alerts.append(alert)
    except FileNotFoundError:
        st.warning(f"Alert file not found: {filepath}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading alerts: {e}")
        return pd.DataFrame()
    
    if not alerts:
        return pd.DataFrame()
    
    df = pd.DataFrame(alerts)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def get_severity_emoji(severity):
    """Get emoji for severity level"""
    emojis = {
        'CRITICAL': '🔴',
        'HIGH': '🟠',
        'MEDIUM': '🟡',
        'LOW': '🔵',
        'INFO': '⚪'
    }
    return emojis.get(severity, '⚫')

def format_confidence(confidence):
    """Format confidence as percentage with color"""
    pct = confidence * 100
    if pct >= 95:
        color = "green"
    elif pct >= 90:
        color = "blue"
    else:
        color = "orange"
    return f":{color}[{pct:.1f}%]"


st.sidebar.title("⚙️ Dashboard Settings")


alert_file = st.sidebar.text_input(
    "Alert File Path",
    value="alerts.jsonl",
    help="Path to the JSONL alert file"
)


auto_refresh = st.sidebar.checkbox("🔄 Auto-Refresh", value=True)
if auto_refresh:
    refresh_interval = st.sidebar.slider(
        "Refresh Interval (seconds)",
        min_value=1,
        max_value=30,
        value=5
    )


st.sidebar.subheader("🔍 Filters")


severity_filter = st.sidebar.multiselect(
    "Severity Levels",
    options=SEVERITY_ORDER,
    default=SEVERITY_ORDER
)


min_confidence = st.sidebar.slider(
    "Minimum Confidence",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.05
)


time_range = st.sidebar.selectbox(
    "Time Range",
    options=['Last 5 min', 'Last 15 min', 'Last 1 hour', 'Last 24 hours', 'All time'],
    index=2
)


show_ambiguous = st.sidebar.checkbox("Show Ambiguous Alerts", value=True)


df = load_alerts(alert_file)


if not df.empty:
    
    df = df[df['severity'].isin(severity_filter)]
    
   
    df = df[df['confidence'] >= min_confidence]
    
    
    if not show_ambiguous:
        df = df[df['attack_id'] != 100]
    
    
    if time_range != 'All time':
        time_map = {
            'Last 5 min': 5,
            'Last 15 min': 15,
            'Last 1 hour': 60,
            'Last 24 hours': 1440
        }
        minutes = time_map[time_range]
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        df = df[df['timestamp'] >= cutoff_time]


st.title("🛡️ Real-Time IDS Alert Dashboard")
st.markdown("---")


if auto_refresh:
    placeholder = st.empty()
    with placeholder.container():
        st.info(f"Auto-refreshing every {refresh_interval} seconds... Last update: {datetime.now().strftime('%H:%M:%S')}")
    time.sleep(0.1)  


if not df.empty:
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="📊 Total Alerts",
            value=f"{len(df):,}",
            delta=None
        )
    
    with col2:
        critical_count = len(df[df['severity'] == 'CRITICAL'])
        st.metric(
            label="🔴 Critical",
            value=critical_count,
            delta=None
        )
    
    with col3:
        high_count = len(df[df['severity'] == 'HIGH'])
        st.metric(
            label="🟠 High",
            value=high_count,
            delta=None
        )
    
    with col4:
        ambiguous_count = len(df[df['attack_id'] == 100])
        st.metric(
            label="🔵 Ambiguous",
            value=ambiguous_count,
            delta=None
        )
    
    with col5:
        avg_confidence = df['confidence'].mean()
        st.metric(
            label="📈 Avg Confidence",
            value=f"{avg_confidence:.1%}",
            delta=None
        )
    
    st.markdown("---")
    
    
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        
        st.subheader("🎯 Attack Type Distribution")
        attack_counts = df['attack_type'].value_counts()
        
        fig_pie = px.pie(
            values=attack_counts.values,
            names=attack_counts.index,
            color=attack_counts.index,
            color_discrete_map=ATTACK_COLORS,
            hole=0.4
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie.update_layout(height=400, showlegend=True)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col_right:
       
        st.subheader("⚠️ Severity Distribution")
        severity_counts = df['severity'].value_counts().reindex(SEVERITY_ORDER, fill_value=0)
        
        fig_bar = px.bar(
            x=severity_counts.index,
            y=severity_counts.values,
            color=severity_counts.index,
            color_discrete_map=SEVERITY_COLORS,
            labels={'x': 'Severity', 'y': 'Count'}
        )
        fig_bar.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    
    st.subheader("📈 Alerts Over Time")
    
   
    df_time = df.set_index('timestamp').resample('1min').size().reset_index()
    df_time.columns = ['timestamp', 'count']
    
    fig_time = px.line(
        df_time,
        x='timestamp',
        y='count',
        labels={'count': 'Alert Count', 'timestamp': 'Time'}
    )
    fig_time.update_traces(line_color='#FF4444', line_width=2)
    fig_time.update_layout(height=300)
    st.plotly_chart(fig_time, use_container_width=True)
    
    
    col_conf1, col_conf2 = st.columns([2, 1])
    
    with col_conf1:
        st.subheader("📊 Confidence Distribution")
        fig_hist = px.histogram(
            df,
            x='confidence',
            nbins=20,
            color_discrete_sequence=['#33B5E5']
        )
        fig_hist.update_layout(
            xaxis_title="Confidence Score",
            yaxis_title="Count",
            height=300
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col_conf2:
        st.subheader("📋 Confidence Stats")
        st.metric("Max", f"{df['confidence'].max():.2%}")
        st.metric("Mean", f"{df['confidence'].mean():.2%}")
        st.metric("Min", f"{df['confidence'].min():.2%}")
        st.metric("Std Dev", f"{df['confidence'].std():.2%}")
    
    st.markdown("---")
    
   
    tab1, tab2, tab3, tab4 = st.tabs(["🚨 Recent Alerts", "🔍 Investigation Queue", "📊 Detailed View", "🔬 Feature Analysis"])
    
    with tab1:
        st.subheader("Recent Alerts")
        
       
        recent_df = df.sort_values('timestamp', ascending=False).head(20)
        
        for idx, alert in recent_df.iterrows():
            severity_emoji = get_severity_emoji(alert['severity'])
            confidence_str = format_confidence(alert['confidence'])
            
            with st.expander(
                f"{severity_emoji} {alert['attack_type']} - {alert['timestamp'].strftime('%H:%M:%S')} - Confidence: {alert['confidence']:.1%}",
                expanded=False
            ):
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.write(f"**Severity:** {alert['severity']}")
                    st.write(f"**Attack Type:** {alert['attack_type']}")
                    st.write(f"**Packet ID:** {alert['id']}")
                    st.write(f"**Timestamp:** {alert['timestamp']}")
                
                with col_b:
                    st.write(f"**Confidence:** {alert['confidence']:.2%}")
                    st.write(f"**Attack ID:** {alert['attack_id']}")
                    st.write(f"**Description:**")
                    st.info(alert['description'])
                
              
                if 'features' in alert and alert['features']:
                    st.write("**Key Features:**")
                    
                    
                    features = alert['features']
                    feat_col1, feat_col2, feat_col3 = st.columns(3)
                    with feat_col1:
                        st.metric("Rate", f"{features.get('Rate', 0):.2f}")
                    with feat_col2:
                        st.metric("Protocol", features.get('Protocol Type', 'N/A'))
                    with feat_col3:
                        st.metric("TTL", features.get('Time_To_Live', 'N/A'))
                    
                   
                    if st.checkbox("View All Features", key=f"features_recent_{idx}"):
                        features_df = pd.DataFrame([alert['features']]).T
                        features_df.columns = ['Value']
                        st.dataframe(features_df, use_container_width=True)
    
    with tab2:
        st.subheader("🔵 Ambiguous Alerts - Investigation Queue")
        
        ambiguous_df = df[df['attack_id'] == 100].sort_values('confidence', ascending=False)
        
        if not ambiguous_df.empty:
            st.write(f"**Total Ambiguous Alerts:** {len(ambiguous_df)}")
            st.write("*Potential zero-day exploits or novel attack patterns*")
            
           
            priority_options = st.radio(
                "Sort by:",
                options=['Confidence (High to Low)', 'Time (Recent First)', 'Time (Oldest First)'],
                horizontal=True
            )
            
            if priority_options == 'Confidence (High to Low)':
                ambiguous_df = ambiguous_df.sort_values('confidence', ascending=False)
            elif priority_options == 'Time (Recent First)':
                ambiguous_df = ambiguous_df.sort_values('timestamp', ascending=False)
            else:
                ambiguous_df = ambiguous_df.sort_values('timestamp', ascending=True)
            
            for idx, alert in ambiguous_df.head(10).iterrows():
                with st.expander(
                    f"🔵 Ambiguous Alert - Confidence: {alert['confidence']:.1%} - {alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}",
                    expanded=False
                ):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Packet ID:** {alert['id']}")
                        st.write(f"**Timestamp:** {alert['timestamp']}")
                        st.write(f"**Confidence:** {alert['confidence']:.2%}")
                    
                    with col2:
                        st.warning("⚠️ Requires Manual Investigation")
                        st.write("**Possible Indicators:**")
                        st.write("• Unknown attack pattern")
                        st.write("• Potential zero-day exploit")
                        st.write("• Novel threat behavior")
                    
                    if 'features' in alert and alert['features']:
                        st.write("**Feature Analysis:**")
                        features = alert['features']
                        
                       
                        col_f1, col_f2, col_f3 = st.columns(3)
                        with col_f1:
                            st.metric("Rate", f"{features.get('Rate', 0):.2f}")
                        with col_f2:
                            st.metric("Protocol", features.get('Protocol Type', 'N/A'))
                        with col_f3:
                            st.metric("TTL", features.get('Time_To_Live', 'N/A'))
                        
                       
                        if st.checkbox("View All Features", key=f"features_ambig_{idx}"):
                            features_df = pd.DataFrame([features]).T
                            features_df.columns = ['Value']
                            st.dataframe(features_df, use_container_width=True)
        else:
            st.success("✅ No ambiguous alerts in current view")
    
    with tab3:
        st.subheader("Detailed Alert Table")
        
        
        display_df = df[['timestamp', 'severity', 'attack_type', 'confidence', 'attack_id', 'id']].copy()
        display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x:.2%}")
        display_df = display_df.sort_values('timestamp', ascending=False)
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "timestamp": st.column_config.DatetimeColumn(
                    "Timestamp",
                    format="YYYY-MM-DD HH:mm:ss"
                ),
                "severity": "Severity",
                "attack_type": "Attack Type",
                "confidence": "Confidence",
                "attack_id": "Attack ID",
                "id": "Packet ID"
            }
        )
        
       
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"ids_alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with tab4:
        st.subheader("🔬 Feature Analysis")
        
        if 'features' in df.columns and not df['features'].isna().all():
          
            all_features = []
            for features_dict in df['features'].dropna():
                if isinstance(features_dict, dict):
                    all_features.append(features_dict)
            
            if all_features:
                features_df = pd.DataFrame(all_features)
                
                st.write("**Feature Statistics:**")
                st.dataframe(features_df.describe(), use_container_width=True)
                
               
                st.write("**Feature Distributions:**")
                selected_feature = st.selectbox(
                    "Select Feature to Analyze",
                    options=sorted(features_df.columns)
                )
                
                if selected_feature:
                    fig_feature = px.histogram(
                        features_df,
                        x=selected_feature,
                        nbins=30,
                        title=f"Distribution of {selected_feature}"
                    )
                    st.plotly_chart(fig_feature, use_container_width=True)
            else:
                st.info("No feature data available")
        else:
            st.info("No feature data available in alerts")

else:
    st.warning("⚠️ No alerts found. Please check the alert file path.")
    st.info("Make sure the IDS simulation is running and generating alerts to alerts.jsonl")


st.markdown("---")
col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    st.caption(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
with col_f2:
    if not df.empty:
        st.caption(f"Showing {len(df):,} alerts")
with col_f3:
    st.caption("🛡️ IDS Alert Dashboard v1.0")


if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()