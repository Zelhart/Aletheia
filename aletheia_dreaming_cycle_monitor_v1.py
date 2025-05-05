#!/usr/bin/env python3
# ===========================================================
# ALETHEIA v1.0 — Dream Cycle / Offline Simulation Monitor
# Purpose: Visualize dreaming sessions and their outcomes.
# ===========================================================

import pandas as pd
import plotly.graph_objects as go

# ---------------------------------
# CONFIGURATION
# ---------------------------------

LOG_FILE = "aletheia_dreaming_log.csv"

# ---------------------------------
# LOAD DATA
# ---------------------------------

try:
    df = pd.read_csv(LOG_FILE)
except FileNotFoundError:
    print("Dreaming log file not found. Ensure 'aletheia_dreaming_log.csv' exists.")
    exit()

if df.empty:
    print("No dreaming data available.")
    exit()

df['dream_start'] = pd.to_datetime(df['dream_start'])
df['dream_end'] = pd.to_datetime(df['dream_end'])

# ---------------------------------
# REQUIRED COLUMNS CHECK
# ---------------------------------

required_cols = ['dream_start', 'dream_end',
                 'seed_memories', 'recombined_memories',
                 'novel_associations', 'error_corrections',
                 'dream_quality_score']

for col in required_cols:
    if col not in df.columns:
        print(f"Missing column: {col}. Please include it in the dreaming log.")
        exit()

# ---------------------------------
# PLOT — Dream Sessions Timeline
# ---------------------------------

fig = go.Figure()

for idx, row in df.iterrows():
    fig.add_trace(go.Scatter(
        x=[row['dream_start'], row['dream_end']],
        y=[idx, idx],
        mode='lines+markers',
        line=dict(color='mediumslateblue', width=4),
        marker=dict(size=8),
        name=f"Dream {idx+1}",
        hovertext=(
            f"Seed Memories: {row['seed_memories']}<br>"
            f"Recombined Memories: {row['recombined_memories']}<br>"
            f"Novel Associations: {row['novel_associations']}<br>"
            f"Error Corrections: {row['error_corrections']}<br>"
            f"Quality Score: {row['dream_quality_score']}"
        ),
        hoverinfo='text'
    ))

# ---------------------------------
# Layout
# ---------------------------------

fig.update_layout(
    title="ALETHEIA Dreaming Cycle Timeline",
    xaxis_title="Time",
    yaxis_title="Dream Session Index",
    yaxis=dict(autorange='reversed'),
    plot_bgcolor='black',
    paper_bgcolor='black',
    font=dict(color='white'),
    hovermode='closest'
)

fig.show()
