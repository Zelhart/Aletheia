# main.py

import logging
from core.cognitive_core import CognitiveCore

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


def run_simulation(cycles: int = 5):
    """
    Initialize the CognitiveCore and run a set number of cognitive cycles.
    """
    logger = logging.getLogger(__name__)

    logger.info("=== Initializing CognitiveCore ===")
    mind = CognitiveCore()
    mind.initialize_cognitive_structures()

    logger.info("=== Starting Simulation ===")

    for _ in range(cycles):
        result = mind.cognitive_cycle()

        # Print high-level cycle summary
        print("\n--- Cognitive Cycle Summary ---")
        print(f"Timestep: {result['timestep']}")
        print(f"Action: {result['action']}")
        print(f"Outcome: {result['outcome']}")
        print(f"Narrative Title: {result['narrative'].title}")
        print(f"Mood: {result['narrative'].mood}")
        print(f"Emotional Theme: {result['narrative'].emotional_theme}")
        print(f"Reflection: {result['narrative'].reflection}")
        print(f"Future Pull: {result['narrative'].future_pull}")
        print("-------------------------------")

    logger.info("=== Simulation Completed ===")


if __name__ == "__main__":
    run_simulation(cycles=5)
