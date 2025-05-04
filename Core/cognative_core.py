import logging
import time
from typing import Optional

from core.perception import SymbolicPerceptionEngine
from core.appraisal import AppraisalEngine
from core.affect import AffectiveModulation
from core.reflection import ReflectionEngine
from core.memory import LucentThreadKeeper
from core.intent import EidonWeaver
from core.action import TelosBloom, ActionProposal
from core.primordium import PrimordiumArc
from core.expression import AitherLoom
from core.learning import EchoMeridian

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CognitiveCore:
    """
    Core cognitive processing system integrating perception, emotion, 
    memory, intention, action, and learning.
    """

    def __init__(self):
        """Initialize the full cognitive architecture."""

        # --- Perception ---
        self.perception_engine = SymbolicPerceptionEngine()

        # --- Appraisal and Affect ---
        self.appraisal_engine = AppraisalEngine(self.perception_engine)
        self.affective_modulation = AffectiveModulation(self.appraisal_engine)

        # --- Reflection & Memory ---
        self.reflection_engine = ReflectionEngine(self.affective_modulation, self.appraisal_engine)
        self.lucent_thread_keeper = LucentThreadKeeper(self.reflection_engine)

        # --- Intention & Action ---
        self.eidon_weaver = EidonWeaver(self.lucent_thread_keeper)
        self.telos_bloom = TelosBloom(self.eidon_weaver)
        self.primordium_arc = PrimordiumArc(self.telos_bloom)
        self.aither_loom = AitherLoom()

        # --- Learning ---
        self.echo_meridian = EchoMeridian(self.appraisal_engine, self.affective_modulation)

        # --- Internal State ---
        self.timestep = 0

        logger.info("Cognitive Core initialized - all systems online.")

    def perceive(self, observations: list):
        """
        Feed new perceptions into the system.

        Args:
            observations: List of symbolic tags observed.
        """
        self.perception_engine.observe(observations)

    def cognitive_cycle(self):
        """
        Execute one complete cognitive cycle.
        """

        self.timestep += 1
        logger.info(f"=== Cognitive Cycle {self.timestep} ===")

        # --- Phase 1: Perception already occurred via perceive() ---

        # --- Phase 2: Appraisal ---
        self.appraisal_engine.appraise_threads()

        # --- Phase 3: Affective Processing ---
        self.affective_modulation.compute_affective_state()

        # --- Phase 4: Reflection ---
        reflection_entry = self.reflection_engine.reflect(self.timestep)
        logger.info(f"Reflection: {reflection_entry.narrative}")

        # --- Phase 5: Memory Weaving ---
        self.lucent_thread_keeper.weave_threads()

        # --- Phase 6: Intent Formation ---
        self.eidon_weaver.form_intents()

        # --- Phase 7: Action Planning ---
        self.telos_bloom.generate_actions()
        self.primordium_arc.commit_action()

        # --- Phase 8: Action Execution ---
        action_result = self.primordium_arc.execute_action()

        # --- Phase 9: Expression ---
        if self.primordium_arc.current_action:
            self.aither_loom.express(self.primordium_arc.current_action)

        # --- Phase 10: Outcome Evaluation and Learning ---
        if self.primordium_arc.current_action:
            outcome = self._evaluate_outcome(self.primordium_arc.current_action, action_result)
            self.echo_meridian.observe_outcome(self.primordium_arc.current_action, outcome)
        else:
            outcome = None

        logger.info(f"=== End of Cycle {self.timestep} ===")

        return {
            "timestep": self.timestep,
            "action": self.primordium_arc.current_action,
            "outcome": outcome
        }

    def _evaluate_outcome(self, action: ActionProposal, result: Optional[dict]) -> str:
        """
        Evaluates the result of an executed action.

        Args:
            action: Action executed.
            result: Result details from execution.

        Returns:
            Outcome string: 'positive', 'neutral', or 'negative'.
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

    def simulate_outcome(self, action: ActionProposal) -> str:
        """
        Placeholder for testing outcomes without environment integration.
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
