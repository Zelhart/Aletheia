# core/codex.py

import logging
from dataclasses import dataclass
from typing import List, Optional

from core.reflection import ReflectionEntry
from core.affect import AffectiveState
from core.intent import Intent

logger = logging.getLogger(__name__)


@dataclass
class CodexCrystal:
    """
    Represents a crystallized snapshot of the agent's state in a cognitive cycle.
    Serves as the structured memory node for narrative generation, review, and learning.
    """
    timestep: int
    dominant_emotion: str
    key_motifs: List[str]
    affective_state: AffectiveState
    top_intents: List[Intent]

    def __repr__(self):
        return (f"CodexCrystal(Timestep={self.timestep}, Emotion='{self.dominant_emotion}', "
                f"Motifs={self.key_motifs}, Intents={len(self.top_intents)})")


class Codex:
    """
    Builds Codex Crystals each cycle by aggregating data from reflection, affect, and intents.
    """

    def __init__(self):
        self.crystals: List[CodexCrystal] = []
        logger.info("Codex initialized. Ready to crystallize meaning each cycle.")

    def crystallize(self,
                    timestep: int,
                    reflection: ReflectionEntry,
                    affective_state: AffectiveState,
                    top_intents: Optional[List[Intent]] = None) -> CodexCrystal:
        """
        Creates a new Codex Crystal from the current cognitive data.

        Args:
            timestep: Current cognitive cycle timestep.
            reflection: The ReflectionEntry for this cycle.
            affective_state: The computed AffectiveState.
            top_intents: Top intents generated for this cycle.

        Returns:
            CodexCrystal: The structured memory snapshot.
        """

        if top_intents is None:
            top_intents = []

        crystal = CodexCrystal(
            timestep=timestep,
            dominant_emotion=reflection.dominant_emotion,
            key_motifs=reflection.symbolic_motifs,
            affective_state=affective_state,
            top_intents=top_intents
        )

        self.crystals.append(crystal)
        logger.info(f"Codex crystallized new memory: {crystal}")

        return crystal

    def review_recent(self, limit: int = 3) -> List[CodexCrystal]:
        """Returns the most recent Codex Crystals for review."""
        return self.crystals[-limit:] if self.crystals else []
