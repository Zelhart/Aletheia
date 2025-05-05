#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cognitive Core for Aletheia
===========================
Integrates appraisal, affect modulation, reflection,
intent generation, action planning, reflexive response,
execution, and learning.
"""

import logging
from core.appraisal import AppraisalEngine
from core.affect import AffectiveModulation
from core.action import ActionProposal
from core.intent import Intent
from core.codex import CodexCrystal
from core.reflection import ReflectionEngine
from core.learning import EchoMeridian
from core.narrative import MythopoeticNarrator
from core.perception import LucentThreadKeeper
from core.eidon import EidonWeaver
from core.telos import TelosBloom
from core.primordium import PrimordiumArc
from core.expression import AitherLoom
from core.reflex_system import ReflexSystem
from core.actuator_bridge import ActuatorBridge

logger = logging.getLogger(__name__)

class CognitiveCore:
    """
    Central orchestrator for Aletheia's cognitive cycle.
    """

    def __init__(self):
        logger.info("Initializing Cognitive Core...")

        self.appraisal_engine = AppraisalEngine()
        self.affective_modulation = AffectiveModulation()
        self.reflection_engine = ReflectionEngine()
        self.lucent_thread_keeper = LucentThreadKeeper()
        self.eidon_weaver = EidonWeaver()
        self.telos_bloom = TelosBloom()
        self.primordium_arc = PrimordiumArc()
        self.aither_loom = AitherLoom()

        # Learning and narrative layers
        self.echo_meridian = EchoMeridian(
            self.appraisal_engine,
            self.affective_modulation,
            learning_rate=0.05
        )
        self.narrator = MythopoeticNarrator()

        # Reflex and action layers
        self.reflex_system = ReflexSystem(self.affective_modulation)
        self.actuator_bridge = ActuatorBridge()

        self.timestep = 0
        logger.info("Cognitive Core initialized — all systems operational.")

    def cognitive_cycle(self):
        """
        Executes one full cognitive cycle.
        Returns summary data for monitoring.
        """
        self.timestep += 1
        logger.info(f"=== Cognitive Cycle {self.timestep} ===")

        # ---- Phase 1: Appraisal ----
        self.appraisal_engine.appraise_threads()

        # ---- Phase 2: Affect Update ----
        self.affective_modulation.compute_affective_state()

        # ---- Phase 3: Reflex Check ----
        reflex_action = self.reflex_system.check_reflexes()

        if reflex_action:
            logger.info("Reflex action selected.")
            selected_action = reflex_action

        else:
            # ---- Phase 4: Reflection ----
            reflection_entry = self.reflection_engine.reflect(self.timestep)
            logger.info(f"Reflection: {reflection_entry.narrative}")

            # ---- Phase 5: Memory ----
            self.lucent_thread_keeper.weave_threads()

            # ---- Phase 6: Intent Formation ----
            self.eidon_weaver.form_intents()

            # ---- Phase 7: Action Planning ----
            self.telos_bloom.generate_actions()
            self.primordium_arc.commit_action()
            selected_action = self.primordium_arc.current_action

            if selected_action:
                logger.info(f"Deliberate action selected: {selected_action.description}")
            else:
                logger.info("No deliberate action selected.")
                # Default fallback action if none planned
                selected_action = ActionProposal(
                    description="Idle Observation",
                    intent=Intent(motif="observe", strength=0.2),
                    urgency=0.2
                )

        # ---- Phase 8: Execution ----
        result = self.actuator_bridge.execute(selected_action)

        # ---- Phase 9: Learning ----
        outcome = self._evaluate_outcome(selected_action, result)
        self.echo_meridian.observe_outcome(selected_action, outcome)

        # ---- Phase 10: Narrative Generation ----
        crystal = CodexCrystal.from_cycle(
            timestep=self.timestep,
            appraisal=self.appraisal_engine,
            affect=self.affective_modulation,
            intents=self.eidon_weaver.top_intents
        )
        narrative = self.narrator.compose_narrative(crystal)

        logger.info(f"Narrative Summary: {narrative.title} | Mood: {narrative.mood}")

        logger.info(f"=== End Cycle {self.timestep} ===\n")

        return {
            "timestep": self.timestep,
            "action": selected_action.description,
            "outcome": outcome,
            "narrative_title": narrative.title,
            "narrative_mood": narrative.mood
        }

    def _evaluate_outcome(self, action: ActionProposal, result: dict) -> str:
        """
        Analyzes action results to classify the outcome.
        """
        if not result:
            logger.warning("No result data available for outcome evaluation.")
            return "neutral"

        if result.get("success", False):
            return "positive"
        elif result.get("partial_success", False):
            return "neutral"
        else:
            return "negative"
