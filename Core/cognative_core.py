# cognitive_core.py

import logging
import random

from core.appraisal import AppraisalEngine
from core.affect import AffectiveModulation
from core.reflection import ReflectionEngine
from core.memory import LucentThreadKeeper
from core.intent import EidonWeaver
from core.action import TelosBloom, PrimordiumArc, ActionProposal
from core.expression import AitherLoom
from core.learning import EchoMeridian
from core.codex import Codex
from narrative.mythopoetic_narrative import MythopoeticNarrator

logger = logging.getLogger(__name__)


class CognitiveCore:
    """
    Core cognitive processing system integrating perception, emotion,
    memory, intention, action, narrative, and learning.
    """

    def __init__(self):
        """Initialize the full cognitive architecture."""

        # === Core Modules ===
        self.appraisal_engine = AppraisalEngine()
        self.affective_modulation = AffectiveModulation(self.appraisal_engine)
        self.reflection_engine = ReflectionEngine(self.affective_modulation, self.appraisal_engine)
        self.lucent_thread_keeper = LucentThreadKeeper(self.reflection_engine)
        self.eidon_weaver = EidonWeaver(self.lucent_thread_keeper)
        self.telos_bloom = TelosBloom(self.eidon_weaver)
        self.primordium_arc = PrimordiumArc(self.telos_bloom)
        self.aither_loom = AitherLoom()

        # === Learning and Memory ===
        self.echo_meridian = EchoMeridian(self.appraisal_engine, self.affective_modulation)
        self.codex = Codex()

        # === Narrative Layer ===
        self.narrator = MythopoeticNarrator()

        # Internal state
        self.timestep = 0

        logger.info("Cognitive Core fully initialized — all systems operational.")

    def initialize_cognitive_structures(self):
        """Create initial motifs and threads for the system."""
        self.appraisal_engine.mythos_forge.create_motif("Vital Surge", 0.8)
        self.appraisal_engine.mythos_forge.create_motif("Core Tension", 0.6)
        self.appraisal_engine.mythos_forge.create_motif("Rising Flame", 0.7)
        self.appraisal_engine.mythos_forge.create_motif("Subtle Whisper", 0.4)

        self.appraisal_engine.mythos_forge.create_thread(
            "Inner Growth", ["Vital Surge", "Rising Flame"]
        )
        self.appraisal_engine.mythos_forge.create_thread(
            "Conflict Resolution", ["Core Tension", "Subtle Whisper"]
        )

        logger.info("Cognitive structures initialized with motifs and threads.")

    def cognitive_cycle(self):
        """
        Execute one complete cognitive cycle through all subsystems.
        This implements the full perception-cognition-action-learning loop.
        """
        self.timestep += 1
        logger.info(f"=== Starting Cognitive Cycle {self.timestep} ===")

        # --- 1. Appraisal ---
        self.appraisal_engine.appraise_threads()

        # --- 2. Emotional Processing ---
        self.affective_modulation.compute_affective_state()

        # --- 3. Reflection ---
        reflection_entry = self.reflection_engine.reflect(self.timestep)
        logger.info(f"Reflection: {reflection_entry.narrative}")

        # --- 4. Memory Formation ---
        self.lucent_thread_keeper.weave_threads()

        # --- 5. Intent Formation ---
        self.eidon_weaver.form_intents()

        # --- 6. Action Selection ---
        self.telos_bloom.generate_actions()
        self.primordium_arc.commit_action()

        # --- 7. Action Execution ---
        action_result = self.primordium_arc.execute_action()

        # --- 8. Expression ---
        if self.primordium_arc.current_action:
            self.aither_loom.express(self.primordium_arc.current_action)

        # --- 9. Codex Crystalization ---
        crystal = self.codex.crystallize(
            timestep=self.timestep,
            reflection=reflection_entry,
            affective_state=self.affective_modulation.current_affective_state,
            top_intents=self.eidon_weaver.get_top_intents()
        )

        # --- 10. Mythopoetic Narrative ---
        narrative = self.narrator.compose_narrative(crystal)
        logger.info(f"Narrative Summary: {narrative.title} | Mood: {narrative.mood}")

        # --- 11. Outcome and Learning ---
        if self.primordium_arc.current_action:
            outcome = self._simulate_outcome(self.primordium_arc.current_action)
            self.echo_meridian.observe_outcome(
                self.primordium_arc.current_action, outcome
            )
        else:
            outcome = None

        logger.info(f"=== Completed Cognitive Cycle {self.timestep} ===")

        return {
            "timestep": self.timestep,
            "action": self.primordium_arc.current_action,
            "outcome": outcome,
            "narrative": narrative
        }

    def _simulate_outcome(self, action: ActionProposal) -> str:
        """
        Temporary method to simulate action outcomes for testing purposes.
        Higher urgency increases the chance of a positive result.
        """
        urgency_factor = min(action.urgency * 0.2, 0.2)
        roll = random.random()

        if roll < (0.4 + urgency_factor):
            return "positive"
        elif roll < 0.7:
            return "neutral"
        else:
            return "negative"
