#!/usr/bin/env python3
# ===========================================================
# ALETHEIA v1.0 — Heartbeat & Memory Echo Monitor
# Purpose: Real-time logging and narrative memory echo.
# ===========================================================

import csv
from datetime import datetime
import os

class ALETHEIA_HeartbeatLogger:
    def __init__(self, agent, log_file="aletheia_heartbeat_log.csv"):
        self.agent = agent
        self.log_file = log_file

        self.columns = [
            'timestamp', 'stimulus', 'pressure', 'coherence_pressure',
            'valence', 'fear', 'desire', 'sorrow', 'contradiction',
            'memory_count', 'dream_output'
        ]

        # Initialize CSV file
        self._initialize_log()

    def _initialize_log(self):
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.columns)

    def log_event(self, stimulus, contradiction, dream_output):
        """Log the agent's state after perceiving a stimulus."""
        state = self.agent

        emotional_state = state.emotion.get_current_state()
        valence = emotional_state.get('joy', 0.0) - emotional_state.get('sorrow', 0.0)

        entry = {
            'timestamp': datetime.now().isoformat(),
            'stimulus': stimulus,
            'pressure': round(state.pressure, 3),
            'coherence_pressure': round(state.coherence.get_pressure(), 3),
            'valence': round(valence, 3),
            'fear': round(emotional_state.get('fear', 0.0), 3),
            'desire': round(emotional_state.get('desire', 0.0), 3),
            'sorrow': round(emotional_state.get('sorrow', 0.0), 3),
            'contradiction': contradiction,
            'memory_count': len(state.memory.autobiographical_story),
            'dream_output': dream_output or ""
        }

        # Write to CSV
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.columns)
            writer.writerow(entry)

        # Console echo
        self._echo_console(entry)

    def _echo_console(self, entry):
        print("\n[Heartbeat] — ALETHEIA Real-Time State")
        for key, value in entry.items():
            print(f"{key}: {value}")
        print("="*40)
