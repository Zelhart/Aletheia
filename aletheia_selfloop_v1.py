#!/usr/bin/env python3
# ===========================================================
# ALETHEIA v1.0 — Self-Loop Analyzer
# Purpose: Visualize certainty decay, meta-confidence, and subjective time perception.
# ===========================================================

import pandas as pd
import plotly.graph_objects as go
import numpy as np

# ---------------------------------
# CONFIGURATION
# ---------------------------------

LOG_FILE = "aletheia_heartbeat_log.csv"

# ---------------------------------
# LOAD DATA
# ---------------------------------

try:
    df = pd.read_csv(LOG_FILE)
except FileNotFoundError:
    print("Log file not found. Ensure 'aletheia_heartbeat_log.csv' exists.")
    exit()

if df.empty:
    print("No data to analyze.")
    exit()

# ---------------------------------
# PREPROCESSING
# ---------------------------------

df['timestamp'] = pd.to_datetime(df['timestamp'])
df.sort_values('timestamp', inplace=True)

# Fallback columns if missing
for col in ['certainty', 'meta_confidence', 'temporal_subjectivity']:
    if col not in df.columns:
        df[col] = np.random.uniform(0.5, 0.9, size=len(df))  # Placeholder values

# Certainty decay logic (if missing, simulate a reasonable trend)
if 'certainty_decay' not in df.columns:
    df['certainty_decay'] = 1 - df['certainty']

# Subjective time speed: <1 = time feels slow, >1 = time feels fast
if 'temporal_subjectivity' not in df.columns:
    df['temporal_subjectivity'] = 1.0

# ---------------------------------
# PLOT SETUP
# ---------------------------------

fig = go.Figure()

# --- Certainty Curve ---
fig.add_trace(go.Scatter(
    x=df['timestamp'], y=df['certainty'],
    mode='lines+markers', name='Certainty',
    line=dict(color='royalblue')
))

# --- Meta-Confidence ---
fig.add_trace(go.Scatter(
    x=df['timestamp'], y=df['meta_confidence'],
    mode='lines+markers', name='Meta-Confidence',
    line=dict(color='orange')
))

# --- Certainty Decay ---
fig.add_trace(go.Scatter(
    x=df['timestamp'], y=df['certainty_decay'],
    mode='lines+markers', name='Certainty Decay',
    line=dict(color='firebrick', dash='dash')
))

# --- Temporal Subjectivity ---
fig.add_trace(go.Scatter(
    x=df['timestamp'], y=df['temporal_subjectivity'],
    mode='lines+markers', name='Subjective Time Speed',
    line=dict(color='green', dash='dot'),
    yaxis='y2'
))

# ---------------------------------
# LAYOUT
# ---------------------------------

fig.update_layout(
    title="ALETHEIA Self-Loop Analyzer — Certainty, Meta-Confidence & Temporal Subjectivity",
    xaxis_title="Time",
    yaxis=dict(
        title="Confidence / Decay (0-1)",
        range=[0, 1.1]
    ),
    yaxis2=dict(
        title="Subjective Time Speed",
        overlaying='y',
        side='right',
        range=[0, 2]
    ),
    hovermode='x unified',
    plot_bgcolor='black',
    paper_bgcolor='black',
    font=dict(color='white')
)

fig.show()
