#!/usr/bin/env python3
# ===========================================================
# ALETHEIA v1.0 — Dream Outcome Influence Map
# Purpose: Visualize how dream-cycle insights shape behavior.
# ===========================================================

import pandas as pd
import plotly.graph_objects as go

# ---------------------------------
# CONFIGURATION
# ---------------------------------

LOG_FILE = "aletheia_dream_outcome_log.csv"

# ---------------------------------
# LOAD DATA
# ---------------------------------

try:
    df = pd.read_csv(LOG_FILE)
except FileNotFoundError:
    print("Dream outcome log not found. Ensure 'aletheia_dream_outcome_log.csv' exists.")
    exit()

if df.empty:
    print("No dream outcome data available.")
    exit()

df['timestamp'] = pd.to_datetime(df['timestamp'])

# ---------------------------------
# REQUIRED COLUMNS CHECK
# ---------------------------------

required_cols = [
    'timestamp', 'dream_event', 'novel_association',
    'influenced_behavior', 'influence_strength', 'persistence_level'
]

for col in required_cols:
    if col not in df.columns:
        print(f"Missing column: {col}. Please include it in the dream outcome log.")
        exit()

# ---------------------------------
# PLOT — Dream Influence Over Time
# ---------------------------------

fig = go.Figure()

# Influence strength over time
fig.add_trace(go.Scatter(
    x=df['timestamp'], y=df['influence_strength'],
    mode='lines+markers', name='Influence Strength',
    line=dict(color='gold')
))

# Persistence level over time
fig.add_trace(go.Scatter(
    x=df['timestamp'], y=df['persistence_level'],
    mode='lines+markers', name='Persistence Level',
    line=dict(color='limegreen')
))

# ---------------------------------
# Event markers — dream events
# ---------------------------------

dream_hover = [
    f"Dream Event: {row['dream_event']}<br>Association: {row['novel_association']}" +
    f"<br>Behavior: {row['influenced_behavior']}" for idx, row in df.iterrows()
]

fig.add_trace(go.Scatter(
    x=df['timestamp'], y=[0] * len(df),
    mode='markers',
    marker=dict(symbol='star', size=12, color='violet'),
    name='Dream Event',
    hovertext=dream_hover,
    hoverinfo='text'
))

# ---------------------------------
# Layout
# ---------------------------------

fig.update_layout(
    title="ALETHEIA Dream-Derived Influence Map",
    xaxis_title="Time",
    yaxis_title="Strength / Persistence",
    yaxis_range=[-0.1, 1.1],
    plot_bgcolor='black',
    paper_bgcolor='black',
    font=dict(color='white'),
    hovermode='x unified'
)

fig.show()
