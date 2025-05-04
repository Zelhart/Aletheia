from typing import Dict, List
import logging

from core.appraisal import AppraisalEngine

logger = logging.getLogger(__name__)

class PerceivedSymbol:
    """
    Represents an external or internal stimulus recognized as meaningful.
    """
    def __init__(self, tag: str, intensity: float):
        self.tag = tag
        self.intensity = max(0.0, min(intensity, 1.0))  # Clamp to [0,1]

    def __repr__(self):
        return f"PerceivedSymbol(Tag={self.tag}, Intensity={self.intensity:.2f})"

class SymbolicPerceptionEngine:
    """
    Processes sensory or contextual inputs into perceived symbols,
    then activates or modulates symbolic motifs accordingly.
    """

    def __init__(self, appraisal_engine: AppraisalEngine):
        self.appraisal_engine = appraisal_engine
        self.perceived_symbols: List[PerceivedSymbol] = []

    def perceive(self, observations: Dict[str, float]) -> None:
        """
        Accepts a dictionary of observations where keys are symbolic tags
        and values are intensities [0,1].

        Example:
            observations = {"Vital Surge": 0.7, "Core Tension": 0.5}
        """
        self.perceived_symbols.clear()

        for tag, intensity in observations.items():
            symbol = PerceivedSymbol(tag, intensity)
            self.perceived_symbols.append(symbol)
            logger.info(f"Perceived symbol: {symbol}")

        # Apply to motifs
        self._activate_motifs()

    def _activate_motifs(self) -> None:
        """
        Updates motif strengths based on perceived symbols.
        """
        for symbol in self.perceived_symbols:
            motif = self.appraisal_engine.mythos_forge.motifs.get(symbol.tag)
            if motif:
                old_strength = motif.strength
                # Simple model: blend current strength with perception intensity
                motif.strength = (motif.strength * 0.8) + (symbol.intensity * 0.2)
                motif.strength = max(0.0, min(motif.strength, 1.0))  # Clamp again
                logger.info(f"Motif '{symbol.tag}' updated: {old_strength:.2f} → {motif.strength:.2f}")
            else:
                logger.warning(f"No motif found for perceived symbol '{symbol.tag}'. Skipping update.")
