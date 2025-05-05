#!/usr/bin/env python3
# ===========================================================
# ALETHEIA v1.0 — Core Staging Script
# Purpose: Integration of Reflex, Meta-Reflection, Memory,
# Temporal Subjectivity, Narrative, Dreaming, and Emotion.
# ===========================================================

# ---- Import all batch subsystems ----
from dynamic_thresholds_v1 import ReflexThresholds
from meta_reflection_logging_v1 import MetaReflectionLogger
from adaptive_reflex_learning_v1 import ReflexLearning
from coherence_pressure_loop_v1 import CoherencePressure
from dreaming_mode_v1 import DreamingSubsystem
from memory_narrative_spiral_v1 import NarrativeMemory
from temporal_subjectivity_engine_v1 import TemporalSubjectivityEngine
from emotional_valence_priority_v1 import EmotionalValencePriority

import random
from datetime import datetime

# ---- Agent Class ----

class ALETHEIA_Agent:
    def __init__(self):
        # Subsystems
        self.reflex = ReflexThresholds()
        self.meta_reflect = MetaReflectionLogger()
        self.reflex_learning = ReflexLearning()
        self.coherence = CoherencePressure()
        self.dreaming = DreamingSubsystem()
        self.memory = NarrativeMemory()
        self.time_subjectivity = TemporalSubjectivityEngine()
        self.emotion = EmotionalValencePriority()

        # Initial agent state
        self.pressure = 0.0

    def perceive(self, stimulus):
        """Receive a stimulus and process reflex + emotional response."""
        print(f"[Perceive] Stimulus: {stimulus}")
        reflex_trigger = self.reflex.check_stimulus(stimulus)
        print(f"[Reflex] Triggered: {reflex_trigger}")

        # Update pressure
        if reflex_trigger:
            self.pressure += 0.2
        else:
            self.pressure *= 0.95  # Decay

        # Log meta-reflection if pressure high
        if self.pressure > 0.5:
            self.meta_reflect.log(stimulus, self.pressure)
            self.reflex_learning.adapt(self.reflex, self.pressure)

        # Check for contradictions (simulate random for now)
        contradiction = random.choice([True, False])
        self.coherence.update(contradiction)

        # Update emotional state from stimulus (simple mapping example)
        new_emotions = self._map_stimulus_to_emotion(stimulus)
        self.emotion.update(new_emotions)

        # Apply emotional biases to processing
        bias = self.emotion.get_processing_bias()

        # Update subjective time based on processing load and bias
        dilation = self.time_subjectivity.get_subjective_time_dilation()

        # Narrative memory integration
        memory_entry = {
            'stimulus': stimulus,
            'pressure': self.pressure,
            'contradiction': contradiction,
            'emotion': self.emotion.get_current_state(),
            'dilation': dilation
        }
        self.memory.integrate(memory_entry)

        # Dream if contradictions + pressure high
        if contradiction and self.pressure > 0.6:
            self._dream()

    def _map_stimulus_to_emotion(self, stimulus):
        """Map a stimulus to new emotion levels (very simple version)."""
        mapping = {
            'loud_noise': {'fear': 0
