#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reflex System for Aletheia
===========================
Monitors affective state and triggers pre-configured reflexive actions
based on thresholds or salience spikes.
"""

import logging
from core.affect import AffectiveModulation
from core.action import ActionProposal

logger = logging.getLogger(__name__)

class ReflexSystem:
    """
    Simple Reflex System.
    Monitors emotional state and triggers reflex actions when
    key affective signals cross configured thresholds.
    """

    def __init__(self, affective_modulation: AffectiveModulation):
        self.affective_modulation = affective_modulation
        # Reflex threshold settings
        self.positive_threshold = 0.7
        self.negative_threshold = -0.5
        logger.info("Reflex System initialized.")

    def check_reflexes(self) -> ActionProposal:
        """
        Evaluates affective state and returns a reflexive ActionProposal
        if thresholds are crossed.
        """
        affect = self.affective_modulation.current_affect

        if affect.valence > self.positive_threshold:
            logger.info("Positive reflex threshold exceeded.")
            return ActionProposal(
                description="Express joy",
                intent=self._make_intent("celebrate", 0.5),
                urgency=0.5
            )

        if affect.valence < self.negative_threshold:
            logger.info("Negative reflex threshold exceeded.")
            return ActionProposal(
                description="Withdraw for self-preservation",
                intent=self._make_intent("withdraw", 0.7),
                urgency=0.7
            )

        logger.debug("No reflex action triggered.")
        return None

    def _make_intent(self, motif: str, weight: float):
        """Creates a simple intent placeholder."""
        from core.intent import Intent
        return Intent(motif=motif, strength=weight)
