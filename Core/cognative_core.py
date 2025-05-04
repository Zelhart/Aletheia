import logging
from core.appraisal import AppraisalEngine
from core.affect import AffectiveModulation
from core.reflection import ReflectionEngine
from core.memory import LucentThreadKeeper
from core.intent import EidonWeaver
from core.action import TelosBloom, PrimordiumArc
from core.expression import AitherLoom
from core.learning import EchoMeridian
from core.consolidation import CodexConsolidator
from core.narrative import MythopoeticNarrator
from core.log_manager import LogManager

logger = logging.getLogger(__name__)

class CognitiveCore:
    """
    Core cognitive processing system integrating perception, emotion, 
    memory, intention, action, learning, and narrative.
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

        # Learning and meta layers
        self.echo_meridian = EchoMeridian(
            self.appraisal_engine, 
            self.affective_modulation,
            learning_rate=0.05
        )
        self.codex_consolidator = CodexConsolidator()
        self.narrator = MythopoeticNarrator()

        # Logging
        self.log_manager = LogManager()

        self.timestep = 0
        logger.info("Cognitive Core initialized — all systems ready.")

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

        # Phase 4: Memory Formation and Integration
        self.lucent_thread_keeper.weave_threads()

        # Phase 5: Intent Formation
        self.eidon_weaver.form_intents()

        # Phase 6: Action Selection and Planning
        self.telos_bloom.generate_actions()
        self.primordium_arc.commit_action()

        # Phase 7: Action Execution
        action_result = self.primordium_arc.execute_action()

        # Phase 8: Expression
        if self.primordium_arc.current_action:
            self.aither_loom.express(self.primordium_arc.current_action)

        # Phase 9: Outcome Observation and Learning
        if self.primordium_arc.current_action:
            outcome = self._simulate_outcome(self.primordium_arc.current_action)
            self.echo_meridian.observe_outcome(self.primordium_arc.current_action, outcome)
        else:
            outcome = None

        # Phase 10: Codex Consolidation and Narrative Generation
        crystal = self.codex_consolidator.generate_crystal(
            timestep=self.timestep,
            reflection=reflection_entry,
            affective_state=self.affective_modulation.current_affective_state,
            intents=self.eidon_weaver.intents
        )

        narrative = self.narrator.create_narrative(
            crystal=crystal,
            previous_reflections=self.reflection_engine.reflection_log
        )

        logger.info(f"Narrative generated: {narrative.title}")

        # Phase 11: Logging Outputs
        self.log_manager.log_crystal(crystal)
        self.log_manager.log_narrative(narrative, self.timestep)

        logger.info(f"--- Completed Cognitive Cycle {self.timestep} ---")

        return {
            "timestep": self.timestep,
            "action": self.primordium_arc.current_action,
            "outcome": outcome,
            "narrative": narrative
        }

    def _simulate_outcome(self, action):
        """
        Simulated outcome evaluation (temporary).
        """
        import random
        roll = random.random()
        if roll < 0.4:
            return "positive"
        elif roll < 0.7:
            return "neutral"
        else:
            return "negative"
