
#!/usr/bin/env python3
# ===========================================================
# ALETHEIA v1.0 — Emotional Valence Processing Bias Visualizer
# Purpose: Show how affective state alters cognitive weighting.
# ===========================================================

import pandas as pd
import plotly.graph_objects as go

# ---------------------------------
# CONFIGURATION
# ---------------------------------

LOG_FILE = "aletheia_valence_bias_log.csv"

# ---------------------------------
# LOAD DATA
# ---------------------------------

try:
    df = pd.read_csv(LOG_FILE)
except FileNotFoundError:
    print("Valence bias log not found. Ensure 'aletheia_valence_bias_log.csv' exists.")
    exit()

if df.empty:
    print("No valence bias data available.")
    exit()

df['timestamp'] = pd.to_datetime(df['timestamp'])

# ---------------------------------
# REQUIRED COLUMNS CHECK
# ---------------------------------

required_cols = [
    'timestamp', 'joy', 'fear', 'desire', 'sorrow',
    'attention_bias', 'memory_access_bias', 'strategy_bias'
]

for col in required_cols:
    if col not in df.columns:
        print(f"Missing column: {col}. Please include it in the valence bias log.")
        exit()

# ---------------------------------
# PLOT — Emotional State Over Time
# ---------------------------------

fig = go.Figure()

# Emotional states
for emotion, color in zip(['joy', 'fear', 'desire', 'sorrow'],
                          ['gold', 'crimson', 'limegreen', 'steelblue']):
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df[emotion],
        mode='lines+markers', name=emotion.capitalize(),
        line=dict(color=color)
    ))

# ---------------------------------
# PLOT — Cognitive Bias Markers
# ---------------------------------

# Attention Bias
fig.add_trace(go.Scatter(
    x=df['timestamp'], y=df['attention_bias'],
    mode='lines', name='Attention Bias',
    line=dict(color='orange', dash='dot')
))

# Memory Access Bias
fig.add_trace(go.Scatter(
    x=df['timestamp'], y=df['memory_access_bias'],
    mode='lines', name='Memory Access Bias',
    line=dict(color='violet', dash='dot')
))

# Strategy Bias
fig.add_trace(go.Scatter(
    x=df['timestamp'], y=df['strategy_bias'],
    mode='lines', name='Strategy Bias',
    line=dict(color='cyan', dash='dot')
))

# ---------------------------------
# Layout
# ---------------------------------

fig.update_layout(
    title="ALETHEIA Emotional Valence & Processing Bias Over Time",
    xaxis_title="Time",
    yaxis_title="Bias / Emotional Intensity",
    yaxis_range=[-1.1, 1.1],
    plot_bgcolor='black',
    paper_bgcolor='black',
    font=dict(color='white'),
    hovermode='x unified'
)

fig.show()
