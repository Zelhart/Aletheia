#!/usr/bin/env python3
# ===========================================================
# Reflex Threshold Adaptation - v1.0
# Purpose: Modulate reflex thresholds based on becoming_pressure
# ===========================================================

def compute_dynamic_threshold(base_threshold, becoming_pressure, adaptation_sensitivity=0.3):
    """
    Computes a new reflex threshold lowered by becoming_pressure.
    Higher pressure = more sensitive reflexes (lower thresholds).
    """
    adjustment = becoming_pressure * adaptation_sensitivity
    new_threshold = base_threshold - adjustment
    return max(-1.0, min(1.0, new_threshold))  # Clamp to valid emotion range

# -----------------------------------------------------------
# EXAMPLE: Using dynamic thresholds in reflex checks
# -----------------------------------------------------------

def check_valence_reflex(agent_valence, base_valence_threshold, becoming_pressure):
    dynamic_valence_threshold = compute_dynamic_threshold(
        base_valence_threshold, becoming_pressure)
    if agent_valence >= dynamic_valence_threshold:
        return True  # Reflex fires
    return False  # No reflex

# -----------------------------------------------------------
# TEST ROUTINE
# -----------------------------------------------------------

def test_dynamic_thresholds():
    print("=== Dynamic Threshold Adjustment Test ===")
    base_threshold = 0.5
    agent_valence = 0.6

    for pressure in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        dynamic_thresh = compute_dynamic_threshold(base_threshold, pressure)
        reflex_fired = check_valence_reflex(agent_valence, base_threshold, pressure)
        print(f"Becoming Pressure: {pressure:.1f} | Dynamic Threshold: {dynamic_thresh:.3f} | Reflex Fired: {reflex_fired}")

if __name__ == "__main__":
    test_dynamic_thresholds()
