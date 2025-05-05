#!/usr/bin/env python3
# ===========================================================
# ALETHEIA v1.0 — Visual Monitor (Live Plotly Dash)
# Purpose: Real-time visualization of emotional and narrative evolution.
# ===========================================================

import pandas as pd
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import datetime

# ---------------------------------
# CONFIGURATION
# ---------------------------------

LOG_FILE = "aletheia_heartbeat_log.csv"
REFRESH_INTERVAL_MS = 5000  # 5 seconds

# ---------------------------------
# DASH APP SETUP
# ---------------------------------

app = Dash(__name__)
app.title = "ALETHEIA Visual Monitor v1.0"

app.layout = html.Div(children=[
    html.H1("ALETHEIA v1.0 — Real-Time Monitor", style={'textAlign': 'center'}),

    dcc.Graph(id='emotion-graph'),
    dcc.Graph(id='coherence-memory-graph'),

    dcc.Interval(
        id='interval-component',
        interval=REFRESH_INTERVAL_MS,
        n_intervals=0
    )
])

# ---------------------------------
# CALLBACK — Update Plots
# ---------------------------------

@app.callback(
    [Output('emotion-graph', 'figure'),
     Output('coherence-memory-graph', 'figure')],
    [Input('interval-component', 'n_intervals')]
)
def update_graph(n):
    try:
        df = pd.read_csv(LOG_FILE)
    except FileNotFoundError:
        return go.Figure(), go.Figure()

    if df.empty:
        return go.Figure(), go.Figure()

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.sort_values('timestamp', inplace=True)

    # --------- Emotion Graph ---------
    fig_emotion = go.Figure()

    fig_emotion.add_trace(go.Scatter(
        x=df['timestamp'], y=df['valence'], mode='lines+markers', name='Valence', line=dict(color='royalblue')
    ))
    fig_emotion.add_trace(go.Scatter(
        x=df['timestamp'], y=df['fear'], mode='lines+markers', name='Fear', line=dict(color='red')
    ))
    fig_emotion.add_trace(go.Scatter(
        x=df['timestamp'], y=df['desire'], mode='lines+markers', name='Desire', line=dict(color='orange')
    ))
    fig_emotion.add_trace(go.Scatter(
        x=df['timestamp'], y=df['sorrow'], mode='lines+markers', name='Sorrow', line=dict(color='purple')
    ))

    # Annotate dreams
    dream_rows = df[df['dream_output'].notna() & (df['dream_output'] != "")]
    for _, row in dream_rows.iterrows():
        fig_emotion.add_annotation(
            x=row['timestamp'], y=row['valence'],
            text="Dream",
            showarrow=True, arrowhead=1, font=dict(color="green")
        )

    fig_emotion.update_layout(
        title="Emotional State Over Time",
        xaxis_title="Time",
        yaxis_title="Emotion Value",
        yaxis=dict(range=[-1.1, 1.1])
    )

    # --------- Coherence & Memory Graph ---------
    fig_coherence = go.Figure()

    fig_coherence.add_trace(go.Scatter(
        x=df['timestamp'], y=df['coherence_pressure'], mode='lines+markers', name='Coherence Pressure',
        line=dict(color='darkgreen')
    ))
    fig_coherence.add_trace(go.Scatter(
        x=df['timestamp'], y=df['memory_count'], mode='lines+markers', name='Memory Count',
        line=dict(color='gold')
    ))

    fig_coherence.update_layout(
        title="Coherence Pressure & Memory Narrative Evolution",
        xaxis_title="Time",
        yaxis_title="Value"
    )

    return fig_emotion, fig_coherence

# ---------------------------------
# RUN SERVER
# ---------------------------------

if __name__ == '__main__':
    app.run_server(debug=True)
