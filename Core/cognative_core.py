from typing import Dict, List, Optional
import logging
import random

from core.appraisal import AppraisalEngine
from core.affect import AffectiveModulation
from core.reflector import ReflectionEngine
from memory.lucent import LucentThreadKeeper
from memory.eidon import EidonWeaver
from memory.telos import TelosBloom
from core.primordium import PrimordiumArc
from core.aither import AitherLoom
from memory.echo_meridian import EchoMeridian
from memory.mythic_codex import MythicCodex

logger = logging.getLogger(__name__)

class CognitiveCore:
    """
    Core cognitive processing system integrating perception, emotion, 
    memory, intention, action, and learning.
    """
    
    def __init__(self, crystallization_interval: int = 5):
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

        # Mythic Codex for crystallizing appraised threads
        self.mythic_codex = MythicCodex(self.appraisal_engine)

        self.timestep = 0
        self.crystallization_interval = crystallization_interval

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
            outcome = self._evaluate_outcome(self.primordium_arc.current_action, action_result)
            self.echo_meridian.observe_outcome(self.primordium_arc.current_action, outcome)

        # Phase 10: Periodic Crystallization into the Mythic Codex
        if self.timestep % self.crystallization_interval == 0:
            self.mythic_codex.crystallize()
            logger.info("Mythic Codex crystallization completed.")

        logger.info(f"--- Completed Cognitive Cycle {self.timestep} ---")
        return {
            "timestep": self.timestep,
            "action": self.primordium_arc.current_action,
            "outcome": outcome if self.primordium_arc.current_action else None
        }

    def _evaluate_outcome(self, action, result: Dict) -> str:
        """
        Evaluates action results to determine outcome classification.
        """
        if not result:
            logger.warning("No result data available for outcome evaluation")
            return "neutral"
        if result.get("success", False):
            return "positive"
        elif result.get("partial_success", False):
            return "neutral"
        else:
            return "negative"
