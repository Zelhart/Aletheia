#!/usr/bin/env python3
# ===========================================================
# ALETHEIA v1.0 — Memory Spiral Viewer
# Purpose: Visualize memory clustering, contradiction density, and narrative evolution.
# ===========================================================

import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ---------------------------------
# CONFIGURATION
# ---------------------------------

LOG_FILE = "aletheia_heartbeat_log.csv"
SPIRAL_TURNS = 8

# ---------------------------------
# LOAD DATA
# ---------------------------------

try:
    df = pd.read_csv(LOG_FILE)
except FileNotFoundError:
    print("Heartbeat log not found. Please ensure 'aletheia_heartbeat_log.csv' exists.")
    exit()

if df.empty:
    print("No data to plot.")
    exit()

# Preprocess timestamps
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.sort_values('timestamp', inplace=True)

# If importance weight or contradiction density are missing, create placeholders
if 'importance_weight' not in df.columns:
    df['importance_weight'] = 0.5  # Default medium
if 'contradiction_density' not in df.columns:
    df['contradiction_density'] = 0.0  # Default none

# ---------------------------------
# SPIRAL COORDINATES
# ---------------------------------

n_points = len(df)
theta = np.linspace(0, SPIRAL_TURNS * 2 * np.pi, n_points)
radii = np.linspace(0.2, 1.0, n_points)

# Convert to Cartesian
df['x'] = radii * np.cos(theta)
df['y'] = radii * np.sin(theta)

# ---------------------------------
# COLOR & SIZE MAPPING
# ---------------------------------

colors = df['contradiction_density']
sizes = 10 + df['importance_weight'] * 30  # Size 10-40

# Dream-enhanced glow marker
df['dream_marker'] = df['dream_output'].apply(lambda x: True if pd.notna(x) and str(x).strip() != '' else False)

# ---------------------------------
# CREATE FIGURE
# ---------------------------------

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df['x'], y=df['y'],
    mode='markers+text',
    marker=dict(
        size=sizes,
        color=colors,
        colorscale='Viridis',
        colorbar=dict(title='Contradiction\nDensity'),
        opacity=0.8,
        line=dict(width=2, color=['white' if d else 'black' for d in df['dream_marker']])
    ),
    text=df['stimulus'],
    hovertext=[
        f"Time: {row['timestamp']}<br>"
        f"Valence: {row['valence']:.2f}, Fear: {row['fear']:.2f}<br>"
        f"Desire: {row['desire']:.2f}, Sorrow: {row['sorrow']:.2f}<br>"
        f"Importance: {row['importance_weight']:.2f}<br>"
        f"Contradiction: {row['contradiction_density']:.2f}<br>"
        f"Dream Output: {row['dream_output'] if row['dream_marker'] else 'None'}"
        for _, row in df.iterrows()
    ],
    hoverinfo='text',
    name='Memory Events'
))

fig.update_layout(
    title="ALETHEIA Memory Spiral — Narrative Coherence Map",
    xaxis=dict(showgrid=False, zeroline=False, visible=False),
    yaxis=dict(showgrid=False, zeroline=False, visible=False),
    plot_bgcolor='black',
    paper_bgcolor='black',
    font=dict(color='white')
)

fig.show()
