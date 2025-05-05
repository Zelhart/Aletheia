#!/usr/bin/env python3
# ===========================================================
# Coherence Pressure Loop - v1.0
# Purpose: Track reflex contradictions & failures, accumulate
# coherence pressure, trigger higher-level reflection/adaptation.
# ===========================================================

from datetime import datetime

# --- CONFIGURABLE PARAMETERS ---
COHERENCE_PRESSURE_INCREMENT = 0.1
COHERENCE_PRESSURE_DECAY = 0.02
COHERENCE_PRESSURE_TRIGGER_THRESHOLD = 0.5

class CoherencePressureMonitor:
    def __init__(self):
        self.coherence_pressure = 0.0
        self.contradiction_log = []

    def log_contradiction(self, reflex_event, context, debug_notes=""):
        timestamp = datetime.now().isoformat()
        log_entry = {
            'timestamp': timestamp,
            'reflex_event': reflex_event,
            'context': context,
            'notes': debug_notes
        }
        print("[CONTRADICTION LOGGED]", log_entry)
        self.contradiction_log.append(log_entry)

        self._increase_pressure()

    def _increase_pressure(self):
        self.coherence_pressure += COHERENCE_PRESSURE_INCREMENT
        self.coherence_pressure = min(1.0, self.coherence_pressure)
        print(f"Coherence Pressure Increased: {self.coherence_pressure:.2f}")

        if self.coherence_pressure >= COHERENCE_PRESSURE_TRIGGER_THRESHOLD:
            self._trigger_coherence_reflection()

    def decay_pressure(self):
        """
        Natural decay of coherence pressure over time.
        """
        self.coherence_pressure = max(
            0.0, self.coherence_pressure - COHERENCE_PRESSURE_DECAY
        )
        print(f"Coherence Pressure Decayed: {self.coherence_pressure:.2f}")

    def _trigger_coherence_reflection(self):
        """
        Placeholder: What happens when coherence pressure crosses the threshold.
        """
        print("[COHERENCE REFLECTION TRIGGERED]")
        print(f"Pressure={self.coherence_pressure:.2f}, Contradictions={len(self.contradiction_log)}")

        # Clear contradiction log after triggering reflection.
        self.contradiction_log = []

        # For now, simply reset pressure (later, this will feed into the Dreaming system)
        self.coherence_pressure = 0.0

# -----------------------------------------------------------
# TEST ROUTINE
# -----------------------------------------------------------

def test_coherence_pressure():
    print("=== Coherence Pressure Loop Test ===")

    pressure_monitor = CoherencePressureMonitor()

    # Simulate reflex contradictions over time
    contradiction_events = [
        ("valence_reflex", "High Valence but failure outcome"),
        ("arousal_reflex", "Arousal response contradicted context"),
        ("dominance_reflex", "Dominance triggered withdrawal"),
        ("valence_reflex", "Positive valence but retreat action"),
        ("arousal_reflex", "Overreaction to neutral stimulus"),
        ("dominance_reflex", "Low dominance but forced assertive action"),
    ]

    for event, context in contradiction_events:
        print(f"\nLogging contradiction: {event}")
        pressure_monitor.log_contradiction(reflex_event=event, context=context)

        # Decay between contradictions (optional)
        pressure_monitor.decay_pressure()

if __name__ == "__main__":
    test_coherence_pressure()
