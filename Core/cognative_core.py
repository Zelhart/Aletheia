from core.appraisal import AppraisalEngine
from core.affect import AffectiveModulation
from core.action import ActionProposal
from core.reflection import ReflectionEngine
from core.memory import LucentThreadKeeper
from core.intent import EidonWeaver
from core.action_selection import TelosBloom
from core.action_execution import PrimordiumArc
from core.expression import AitherLoom
from core.learning import EchoMeridian
from core.mythic_codex import MythicCodex  # NEW

import logging
import random

logger = logging.getLogger(__name__)

class CognitiveCore:
    """
    Core cognitive processing system integrating perception, emotion, 
    memory, intention, action, learning, and now — crystallization.
    """

    def __init__(self):
        """Initialize the cognitive architecture components."""
        # Initialize all cognitive subsystems
        self.appraisal_engine = AppraisalEngine()
        self.affective_modulation = AffectiveModulation()
        self.reflection_engine = ReflectionEngine()
        self.lucent_thread_keeper = LucentThreadKeeper()
        self.eidon_weaver = EidonWeaver()
        self.telos_bloom = TelosBloom()
        self.primordium_arc = PrimordiumArc()
        self.aither_loom = AitherLoom()

        self.echo_meridian = EchoMeridian(
            self.appraisal_engine, 
            self.affective_modulation,
            learning_rate=0.05
        )

        self.mythic_codex = MythicCodex()  # NEW MEMORY LAYER

        self.timestep = 0
        logger.info("Cognitive Core initialized - all systems ready")

    def cognitive_cycle(self):
        """
        Execute one complete cognitive cycle through all subsystems.
        """

        self.timestep += 1
        logger.info(f"--- Starting Cognitive Cycle {self.timestep} ---")

        # Phase 1: Perception and Appraisal
        self.appraisal_engine.appraise_threads()

        # Phase 2: Emotional Processing
        self.affective_modulation.compute_affective_state()

        # Phase 3: Reflection and Meaning-Making
        reflection_entry = self.reflection_engine.reflect(self.timestep)
        logger.info(f"Reflection: {reflection_entry.narrative}")

        # Phase 4: Memory Formation
        self.lucent_thread_keeper.weave_threads()

        # Phase 5: Intent Formation
        self.eidon_weaver.form_intents()

        # Phase 6: Action Selection
        self.telos_bloom.generate_actions()
        self.primordium_arc.commit_action()

        # Phase 7: Action Execution
        action_result = self.primordium_arc.execute_action()

        # Phase 8: Expression
        if self.primordium_arc.current_action:
            self.aither_loom.express(self.primordium_arc.current_action)

        # Phase 9: Outcome Observation and Learning
        if self.primordium_arc.current_action:
            outcome = self._evaluate_outcome(self.primordium_arc.current_action, action_result)
            self.echo_meridian.observe_outcome(self.primordium_arc.current_action, outcome)
        else:
            outcome = None

        # === Phase 10: Thread Crystallization ===
        logger.info("Phase 10: Archiving significant threads to Mythic Codex")
        high_priority_threads = self.appraisal_engine.get_high_priority_threads()
        for thread in high_priority_threads:
            self.mythic_codex.archive_thread(thread)

        logger.info(f"--- Completed Cognitive Cycle {self.timestep} ---")

        return {
            "timestep": self.timestep,
            "action": self.primordium_arc.current_action,
            "outcome": outcome,
            "codex_entries": self.mythic_codex.list_codex()
        }
