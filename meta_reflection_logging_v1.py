#!/usr/bin/env python3
# ===========================================================
# Meta-Reflection Logging - v1.0
# Purpose: When reflexes fire under high becoming pressure,
# trigger meta-reflection logging.
# ===========================================================

from datetime import datetime

# --- CONFIGURABLE PARAMETER ---
META_REFLECTION_PRESSURE_THRESHOLD = 0.5  # Above this, reflection logs

def log_meta_reflection(event_type, becoming_pressure, emotional_state, debug_notes=""):
    """
    Logs a meta-reflection event to console or (future) to memory/log buffer.
    """
    timestamp = datetime.now().isoformat()
    log_entry = {
        'timestamp': timestamp,
        'event': event_type,
        'becoming_pressure': becoming_pressure,
        'emotional_state': emotional_state,
        'notes': debug_notes
    }
    print("[META-REFLECTION]", log_entry)
    # Future: append to memory buffer or persistent log

def handle_reflex_with_meta_reflection(
    agent_valence, dynamic_valence_threshold,
    becoming_pressure, emotional_state
):
    """
    Example reflex handler that also checks for meta-reflection logging.
    """
    if agent_valence >= dynamic_valence_threshold:
        print(f"Reflex Fired: Valence {agent_valence} >= Threshold {dynamic_valence_threshold}")

        if becoming_pressure >= META_REFLECTION_PRESSURE_THRESHOLD:
            log_meta_reflection(
                event_type="Valence Reflex + High Becoming Pressure",
                becoming_pressure=becoming_pressure,
                emotional_state=emotional_state,
                debug_notes="Reflex firing triggered self-reflection impulse."
            )
        return True

    return False

# -----------------------------------------------------------
# TEST ROUTINE
# -----------------------------------------------------------

def test_meta_reflection():
    print("=== Meta-Reflection Logging Test ===")

    test_cases = [
        (0.6, 0.5, 0.3),  # Reflex fires, but low pressure
        (0.7, 0.5, 0.6),  # Reflex fires, high pressure triggers meta-reflection
        (0.4, 0.5, 0.7),  # No reflex, so no reflection
    ]

    for valence, threshold, pressure in test_cases:
        emotional_state = {
            'valence': valence,
            'arousal': 0.2,
            'dominance': 0.1
        }
        print(f"\nTesting valence={valence}, threshold={threshold}, pressure={pressure}")
        handle_reflex_with_meta_reflection(
            agent_valence=valence,
            dynamic_valence_threshold=threshold,
            becoming_pressure=pressure,
            emotional_state=emotional_state
        )

if __name__ == "__main__":
    test_meta_reflection()
