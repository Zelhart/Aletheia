import logging
from core.perception import MythosForge
from core.appraisal import AppraisalEngine
from core.affect import AffectiveModulation
from core.reflection import ReflectionEngine
from core.memory import LucentThreadKeeper
from core.intent import EidonWeaver
from core.action import TelosBloom, PrimordiumArc, AitherLoom
from core.learning import EchoMeridian
from core.codex import CodexConsolidator
from narrative.mythopoetic_narrator import MythopoeticNarrator

logger = logging.getLogger(__name__)


class CognitiveCore:
    """
    Core cognitive processing system integrating perception, emotion,
    memory, intention, action, learning, and now mythopoetic narration.
    """

    def __init__(self):
        """Initialize the full cognitive architecture components."""
        # === Core Cognitive Loop Systems ===
        self.mythos_forge = MythosForge()
        self.appraisal_engine = AppraisalEngine(self.mythos_forge)
        self.affective_modulation = AffectiveModulation(self.appraisal_engine)
        self.reflection_engine = ReflectionEngine(self.affective_modulation, self.appraisal_engine)
        self.lucent_thread_keeper = LucentThreadKeeper(self.reflection_engine)
        self.eidon_weaver = EidonWeaver(self.lucent_thread_keeper)
        self.telos_bloom = TelosBloom(self.eidon_weaver)
        self.primordium_arc = PrimordiumArc(self.telos_bloom)
        self.aither_loom = AitherLoom()

        # === Learning & Meta-Cognition ===
        self.echo_meridian = EchoMeridian(
            self.appraisal_engine,
            self.affective_modulation,
            learning_rate=0.05
        )

        # === Memory Crystallization & Narrative Layer ===
        self.codex_consolidator = CodexConsolidator(
            self.reflection_engine,
            self.lucent_thread_keeper,
            self.eidon_weaver,
            self.affective_modulation
        )
        self.narrator = MythopoeticNarrator()

        self.timestep = 0
        logger.info("Cognitive Core initialized - all systems ready")

    def initialize_cognitive_structures(self):
        """Create initial motifs and threads for the system."""
        # Create some example motifs
        self.mythos_forge.create_motif("Vital Surge", 0.8)
        self.mythos_forge.create_motif("Core Tension", 0.6)
        self.mythos_forge.create_motif("Rising Flame", 0.7)
        self.mythos_forge.create_motif("Subtle Whisper", 0.4)

        # Create example threads
        self.mythos_forge.create_thread("Inner Growth", ["Vital Surge", "Rising Flame"])
        self.mythos_forge.create_thread("Conflict Resolution", ["Core Tension", "Subtle Whisper"])

    def cognitive_cycle(self):
        """
        Executes one complete cognitive cycle:
        Perception → Appraisal → Emotion → Reflection → Memory → Intent → Action → Learning → Narrative.
        """
        self.timestep += 1
        logger.info(f"--- Starting Cognitive Cycle {self.timestep} ---")

        # === Perception and Appraisal ===
        self.appraisal_engine.appraise_threads()

        # === Emotional Processing ===
        self.affective_modulation.compute_affective_state()

        # === Reflection ===
        reflection_entry = self.reflection_engine.reflect(self.timestep)
        logger.info(f"Reflection: {reflection_entry.narrative}")

        # === Memory Formation ===
        self.lucent_thread_keeper.weave_threads()

        # === Intent Formation ===
        self.eidon_weaver.form_intents()

        # === Action Selection and Planning ===
        self.telos_bloom.generate_actions()
        self.primordium_arc.commit_action()

        # === Action Execution ===
        action_result = self.primordium_arc.execute_action()

        # === Expression ===
        if self.primordium_arc.current_action:
            self.aither_loom.express(self.primordium_arc.current_action)

        # === Outcome Observation and Learning ===
        if self.primordium_arc.current_action:
            outcome = self._simulate_outcome(self.primordium_arc.current_action)
            self.echo_meridian.observe_outcome(self.primordium_arc.current_action, outcome)
        else:
            outcome = None

        # === Crystallization ===
        crystal = self.codex_consolidator.consolidate(self.timestep)

        # === Mythopoetic Narrative Generation ===
        narrative = self.narrator.compose_narrative(crystal)

        # === Log and Output ===
        logger.info(f"Narrative: {narrative.title} — {narrative.reflection}")
        logger.info(f"Future Guidance: {narrative.future_pull}")

        logger.info(f"--- Completed Cognitive Cycle {self.timestep} ---")

        return {
            "timestep": self.timestep,
            "action": self.primordium_arc.current_action,
            "outcome": outcome,
            "codex_crystal": crystal,
            "narrative": narrative
        }

    def _simulate_outcome(self, action) -> str:
        """
        Simulated outcome evaluation for development/testing.
        In a real system, this would analyze the real-world result of the action.
        """
        import random
        urgency_factor = min(action.urgency * 0.2, 0.2)
        roll = random.random()

        if roll < (0.4 + urgency_factor):
            return "positive"
        elif roll < 0.7:
            return "neutral"
        else:
            return "negative"
