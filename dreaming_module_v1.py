#!/usr/bin/env python3
# ===========================================================
# Dreaming Mode v1.0
# Purpose: Recombine contradictions & recent experiences
# to seek resolutions, prepare adaptations for wake state.
# ===========================================================

import random
from datetime import datetime

class DreamingSubsystem:
    def __init__(self):
        self.recent_events = []  # List of recent reflex outcomes
        self.contradictions = []  # Contradictions logged by CoherencePressureMonitor
        self.dream_outputs = []

    def log_event(self, event_data):
        """
        Log an event or experience for later recombination.
        Example event_data: {
            'timestamp': datetime,
            'reflex_event': 'valence_reflex',
            'context': 'Positive valence but retreat action',
            'outcome': 'unexpected_failure'
        }
        """
        self.recent_events.append(event_data)
        if len(self.recent_events) > 50:
            self.recent_events.pop(0)  # Keep buffer size manageable

    def receive_contradictions(self, contradiction_log):
        """
        Import contradictions from the coherence pressure monitor.
        """
        self.contradictions.extend(contradiction_log)

    def enter_dream_state(self):
        """
        Run a dreaming cycle: recombine contradictions & recent events.
        """
        print("\n[ENTERING DREAMING MODE]")
        self.dream_outputs.clear()

        seed_events = random.sample(
            self.recent_events, min(3, len(self.recent_events))
        ) if self.recent_events else []

        contradiction_seeds = random.sample(
            self.contradictions, min(2, len(self.contradictions))
        ) if self.contradictions else []

        # --- Phase 1: Recombination ---
        for i in range(max(len(seed_events), len(contradiction_seeds))):
            event = seed_events[i % len(seed_events)] if seed_events else {}
            contradiction = contradiction_seeds[i % len(contradiction_seeds)] if contradiction_seeds else {}

            dream_entry = self._recombine(event, contradiction)
            self.dream_outputs.append(dream_entry)

        # --- Phase 2: Generate Resolutions ---
        resolutions = self._generate_resolutions()

        print("[DREAMING OUTPUT]")
        for res in resolutions:
            print(res)

        return resolutions

    def _recombine(self, event, contradiction):
        """
        Create a hybrid 'dream scenario' by mixing event and contradiction.
        """
        new_scenario = {
            'timestamp': datetime.now().isoformat(),
            'combined_context': f"{event.get('context', 'none')} / {contradiction.get('context', 'none')}",
            'reflex_bias_proposal': random.choice(['increase_valence', 'reduce_arousal', 'boost_dominance', 'adjust_thresholds'])
        }
        return new_scenario

    def _generate_resolutions(self):
        """
        Turn dream scenarios into proposed adaptations.
        """
        resolutions = []
        for dream in self.dream_outputs:
            resolution = {
                'timestamp': dream['timestamp'],
                'proposal': f"Suggest to {dream['reflex_bias_proposal']} for context: {dream['combined_context']}"
            }
            resolutions.append(resolution)
        return resolutions

# -----------------------------------------------------------
# TEST ROUTINE
# -----------------------------------------------------------

def test_dreaming_subsystem():
    dreaming = DreamingSubsystem()

    # --- Simulate logging recent events ---
    sample_events = [
        {'timestamp': datetime.now(), 'reflex_event': 'valence_reflex', 'context': 'Joy but hesitation', 'outcome': 'suboptimal'},
        {'timestamp': datetime.now(), 'reflex_event': 'dominance_reflex', 'context': 'Low dominance during threat', 'outcome': 'failure'},
        {'timestamp': datetime.now(), 'reflex_event': 'arousal_reflex', 'context': 'High arousal but passive action', 'outcome': 'mismatch'}
    ]

    for event in sample_events:
        dreaming.log_event(event)

    # --- Simulate contradictions from pressure monitor ---
    sample_contradictions = [
        {'timestamp': datetime.now().isoformat(), 'reflex_event': 'valence_reflex', 'context': 'Contradicted positive expectation'},
        {'timestamp': datetime.now().isoformat(), 'reflex_event': 'arousal_reflex', 'context': 'Overreaction to minor stimulus'}
    ]

    dreaming.receive_contradictions(sample_contradictions)

    # --- Run Dreaming Cycle ---
    dreaming.enter_dream_state()

if __name__ == "__main__":
    test_dreaming_subsystem()
