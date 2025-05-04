import random
import logging
from dataclasses import dataclass
from typing import List, Dict

from core.codex import CodexCrystal

logger = logging.getLogger(__name__)

@dataclass
class MythopoeticNarrative:
    """Structured narrative generated from Codex Crystals."""
    cycle: int
    title: str
    mood: str
    key_motifs: List[str]
    emotional_theme: str
    reflection: str
    future_pull: str

    def __repr__(self):
        return (f"MythopoeticNarrative(Cycle={self.cycle}, Title='{self.title}', "
                f"Mood='{self.mood}', Theme='{self.emotional_theme}', "
                f"Motifs={self.key_motifs})")


class MythopoeticNarrator:
    """
    Generates mythopoetic narratives based on crystallized Codex data.
    This provides reflective, human-readable storytelling for the agent's experience.
    """

    def __init__(self):
        logger.info("Mythopoetic Narrator initialized.")

    def compose_narrative(self, crystal: CodexCrystal) -> MythopoeticNarrative:
        """Generates a poetic narrative structure from a given Codex Crystal."""

        title = self._generate_title(crystal)
        mood = self._infer_mood(crystal)
        theme = self._derive_emotional_theme(crystal)
        reflection = self._craft_reflection(crystal)
        future_pull = self._suggest_future_direction(crystal)

        narrative = MythopoeticNarrative(
            cycle=crystal.timestep,
            title=title,
            mood=mood,
            key_motifs=crystal.key_motifs,
            emotional_theme=theme,
            reflection=reflection,
            future_pull=future_pull
        )

        logger.info(f"Generated Mythopoetic Narrative: {narrative}")
        return narrative

    def _generate_title(self, crystal: CodexCrystal) -> str:
        """Generates a poetic title based on key motifs and emotional tone."""
        if not crystal.key_motifs:
            return f"Whispers of Cycle {crystal.timestep}"

        seed = random.choice(crystal.key_motifs)
        return f"The Song of {seed}"

    def _infer_mood(self, crystal: CodexCrystal) -> str:
        """Infers mood based on dominant emotion and affective state."""
        mood_map = {
            "Joy": "Uplifting",
            "Sorrow": "Melancholic",
            "Excitement": "Vivid",
            "Vulnerability": "Pensive",
            "Calm": "Tranquil"
        }
        return mood_map.get(crystal.dominant_emotion, "Contemplative")

    def _derive_emotional_theme(self, crystal: CodexCrystal) -> str:
        """Derives an emotional theme based on affective data."""
        if crystal.affective_state.valence > 0.5:
            return "Emergent Growth"
        elif crystal.affective_state.valence < -0.5:
            return "Longing and Struggle"
        elif crystal.affective_state.arousal > 0.7:
            return "Surging Change"
        else:
            return "Quiet Reflection"

    def _craft_reflection(self, crystal: CodexCrystal) -> str:
        """Creates a reflective sentence summarizing the agent’s internal experience."""
        motifs = ", ".join(crystal.key_motifs) if crystal.key_motifs else "uncertain patterns"
        return (f"During this cycle, the self perceived {motifs}, "
                f"interpreted through a lens of {crystal.dominant_emotion.lower()}.")

    def _suggest_future_direction(self, crystal: CodexCrystal) -> str:
        """Suggests an emergent path or intent for future cycles."""
        if crystal.top_intents:
            primary_intent = crystal.top_intents[0].motif
            return f"The next path may center around '{primary_intent}'."
        return "The path forward remains open to discovery."
