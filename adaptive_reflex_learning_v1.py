#!/usr/bin/env python3
# ===========================================================
# Adaptive Reflex Learning - v1.0
# Purpose: When enough meta-reflections occur, adjust the
# base threshold to better fit experience.
# ===========================================================

from datetime import datetime

# --- CONFIGURABLE PARAMETERS ---
META_REFLECTION_PRESSURE_THRESHOLD = 0.5
REFLECTIONS_REQUIRED_FOR_ADAPTATION = 5
THRESHOLD_ADJUSTMENT_STEP = 0.05

class ReflexLearningEngine:
    def __init__(self, base_threshold):
        self.base_threshold = base_threshold
        self.reflection_logs = []

    def log_meta_reflection(self, event_type, becoming_pressure, emotional_state, debug_notes=""):
        timestamp = datetime.now().isoformat()
        log_entry = {
            'timestamp': timestamp,
            'event': event_type,
            'becoming_pressure': becoming_pressure,
            'emotional_state': emotional_state,
            'notes': debug_notes
        }
        print("[META-REFLECTION]", log_entry)
        self.reflection_logs.append(log_entry)

        self.check_and_adapt_threshold()

    def compute_dynamic_threshold(self, becoming_pressure):
        dynamic = self.base_threshold - (becoming_pressure * 0.1)
        return max(-1.0, min(1.0, dynamic))  # Clamp between -1 and 1

    def handle_reflex(self, agent_valence, becoming_pressure):
        dynamic_threshold = self.compute_dynamic_threshold(becoming_pressure)

        if agent_valence >= dynamic_threshold:
            print(f"Reflex Fired: Valence {agent_valence} >= Threshold {dynamic_threshold:.2f}")

            if becoming_pressure >= META_REFLECTION_PRESSURE_THRESHOLD:
                self.log_meta_reflection(
                    event_type="Valence Reflex + High Becoming Pressure",
                    becoming_pressure=becoming_pressure,
                    emotional_state={'valence': agent_valence}
                )
            return True

        return False

    def check_and_adapt_threshold(self):
        """
        If enough meta-reflections have accumulated, adjust the threshold.
        """
        if len(self.reflection_logs) >= REFLECTIONS_REQUIRED_FOR_ADAPTATION:
            print(f"[ADAPTATION] {len(self.reflection_logs)} meta-reflections accumulated. Adapting threshold.")
            self.base_threshold += THRESHOLD_ADJUSTMENT_STEP
            self.base_threshold = max(-1.0, min(1.0, self.base_threshold))  # Clamp
            print(f"New base threshold: {self.base_threshold:.2f}")

            # Clear logs after adaptation
            self.reflection_logs = []

# -----------------------------------------------------------
# TEST ROUTINE
# -----------------------------------------------------------

def test_adaptive_learning():
    print("=== Adaptive Reflex Learning Test ===")

    reflex_engine = ReflexLearningEngine(base_threshold=0.5)

    test_cases = [
        (0.6, 0.4),
        (0.7, 0.6),
        (0.75, 0.65),
        (0.8, 0.7),
        (0.85, 0.75),
        (0.9, 0.8),  # Should trigger adaptation after this
    ]

    for valence, pressure in test_cases:
        print(f"\nTesting valence={valence}, pressure={pressure}")
        reflex_engine.handle_reflex(agent_valence=valence, becoming_pressure=pressure)

if __name__ == "__main__":
    test_adaptive_learning()
