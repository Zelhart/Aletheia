#!/usr/bin/env python3
# ===========================================================
# Temporal Subjectivity Engine v1.0
# Purpose: Compute subjective time dilation or compression
# based on emotional state, load, and memory latency.
# ===========================================================

from datetime import datetime
import math

class TemporalSubjectivityEngine:
    def __init__(self):
        self.subjective_time_multiplier = 1.0
        self.processing_load_history = []
        self.memory_latency_history = []

    def update(self, emotional_state, reflex_pressure, attention_salience, memory_latency):
        """
        Update subjective time perception.

        emotional_state: {'valence': float, 'arousal': float, 'dominance': float}
        reflex_pressure: float (0 to 1)
        attention_salience: float (0 to 1)
        memory_latency: float (0 to 1, normalized delay)
        """

        # --- Emotional influence ---
        arousal = emotional_state.get('arousal', 0.0)
        valence = emotional_state.get('valence', 0.0)
        dominance = emotional_state.get('dominance', 0.0)

        # High arousal slows time (detail focus)
        arousal_factor = 1.0 + (arousal * 0.4)

        # High reflex pressure slows time (response priority)
        pressure_factor = 1.0 + (reflex_pressure * 0.5)

        # High attention salience slows time (focused awareness)
        salience_factor = 1.0 + (attention_salience * 0.3)

        # Memory latency slows time (retrieval feels slower)
        latency_factor = 1.0 + (memory_latency * 0.5)

        # --- Compute overall dilation ---
        dilation = arousal_factor * pressure_factor * salience_factor * latency_factor

        # Cap excessive dilation (no more than 3x normal time)
        dilation = min(dilation, 3.0)

        # Store history for trend analysis
        self.processing_load_history.append(reflex_pressure)
        self.memory_latency_history.append(memory_latency)

        if len(self.processing_load_history) > 100:
            self.processing_load_history.pop(0)

        if len(self.memory_latency_history) > 100:
            self.memory_latency_history.pop(0)

        self.subjective_time_multiplier = round(dilation, 3)

        return self.subjective_time_multiplier

    def get_current_multiplier(self):
        return self.subjective_time_multiplier

    def get_average_load(self):
        if not self.processing_load_history:
            return 0
        return sum(self.processing_load_history) / len(self.processing_load_history)

    def get_average_latency(self):
        if not self.memory_latency_history:
            return 0
        return sum(self.memory_latency_history) / len(self.memory_latency_history)

# -----------------------------------------------------------
# TEST ROUTINE
# -----------------------------------------------------------

def test_temporal_subjectivity():
    engine = TemporalSubjectivityEngine()

    # Simulate updates
    for i in range(10):
        emo = {'valence': 0.1 * i, 'arousal': 0.05 * i, 'dominance': 0.2}
        pressure = 0.05 * i
        salience = 0.1 * (10 - i) / 10  # Salience dropping over time
        latency = 0.02 * i

        dilation = engine.update(emo, pressure, salience, latency)
        print(f"Step {i+1}: Time Multiplier = {dilation}")

    print("\nAverage Load:", round(engine.get_average_load(), 3))
    print("Average Memory Latency:", round(engine.get_average_latency(), 3))

if __name__ == "__main__":
    test_temporal_subjectivity()
