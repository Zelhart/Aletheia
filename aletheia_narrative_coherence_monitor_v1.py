#!/usr/bin/env python3
# ===========================================================
# ALETHEIA v1.0 — Narrative Coherence Spiral Monitor
# Purpose: Visualize memory integration, weighting, and contradictions.
# ===========================================================

import pandas as pd
import plotly.graph_objects as go

# ---------------------------------
# CONFIGURATION
# ---------------------------------

LOG_FILE = "aletheia_narrative_log.csv"

# ---------------------------------
# LOAD DATA
# ---------------------------------

try:
    df = pd.read_csv(LOG_FILE)
except FileNotFoundError:
    print("Narrative log file not found. Ensure 'aletheia_narrative_log.csv' exists.")
    exit()

if df.empty:
    print("No narrative data available.")
    exit()

df['timestamp'] = pd.to_datetime(df['timestamp'])

# ---------------------------------
# REQUIRED COLUMNS CHECK
# ---------------------------------

required_cols = ['timestamp', 'memory_id', 'category',
                 'importance_weight', 'identity_alignment',
                 'contradiction_level', 'integration_status']

for col in required_cols:
    if col not in df.columns:
        print(f"Missing column: {col}. Please include it in the narrative log.")
        exit()

# ---------------------------------
# PLOT — Memory Weight & Alignment Over Time
# ---------------------------------

fig = go.Figure()

# Plot Importance Weight
fig.add_trace(go.Scatter(
    x=df['timestamp'],
    y=df['importance_weight'],
    mode='lines+markers',
    name='Importance Weight',
    line=dict(color='gold'),
    text=df['memory_id']
))

# Plot Identity Alignment
fig.add_trace(go.Scatter(
    x=df['timestamp'],
    y=df['identity_alignment'],
    mode='lines+markers',
    name='Identity Alignment',
    line=dict(color='dodgerblue'),
    text=df['memory_id']
))

# Plot Contradiction Level
fig.add_trace(go.Scatter(
    x=df['timestamp'],
    y=df['contradiction_level'],
    mode='lines+markers',
    name='Contradiction Level',
    line=dict(color='firebrick'),
    text=df['memory_id']
))

# ---------------------------------
# EVENT MARKERS — Integration Status
# ---------------------------------

statuses = df['integration_status'].unique()

status_colors = {
    'integrated': 'green',
    'suppressed': 'gray',
    'forcing_identity_shift': 'purple'
}

for status in statuses:
    filtered = df[df['integration_status'] == status]
    if not filtered.empty:
        fig.add_trace(go.Scatter(
            x=filtered['timestamp'],
            y=filtered['importance_weight'],
            mode='markers',
            marker=dict(size=10, color=status_colors.get(status, 'white'), symbol='circle'),
            name=f"Status: {status}",
            text=filtered['memory_id'],
            hoverinfo='text'
        ))

# ---------------------------------
# Layout
# ---------------------------------

fig.update_layout(
    title="ALETHEIA Narrative Coherence Spiral",
    xaxis_title="Time",
    yaxis_title="Normalized Values (0 to 1.5)",
    yaxis_range=[0, 1.5],
    plot_bgcolor='black',
    paper_bgcolor='black',
    font=dict(color='white'),
    hovermode='x unified'
)

fig.show()
