import logging
from core.codex import CodexCrystal

logger = logging.getLogger(__name__)

class LearningInspector:
    """
    Analyzes outcome history from Echo Meridian and surfaces learning patterns.
    """

    def __init__(self, echo_meridian):
        self.echo_meridian = echo_meridian
        logger.info("LearningInspector initialized.")

    def summarize_motif_performance(self):
        """Prints success metrics for all motifs observed."""
        unique_motifs = set(record.motif for record in self.echo_meridian.history)

        summary = {}
        for motif in unique_motifs:
            stats = self.echo_meridian.get_motif_success_rate(motif)
            summary[motif] = stats

            print(f"Motif '{motif}': {stats['success_rate']*100:.1f}% success over {stats['total_attempts']} attempts")

        return summary

    def latest_learning_reflection(self) -> str:
        """Generates a simple text reflection summarizing recent learning."""
        if not self.echo_meridian.history:
            return "No learning data recorded yet."

        recent = self.echo_meridian.history[-1]
        return (f"Most recent outcome: Motif '{recent.motif}' had a '{recent.outcome}' outcome "
                f"at urgency {recent.urgency:.2f}.")
