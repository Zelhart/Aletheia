from core.mythic_codex import MythicCodex
import logging

logger = logging.getLogger(__name__)

class CodexReporter:
    """
    Utility for summarizing and reporting the contents of the Mythic Codex.
    """

    def __init__(self, mythic_codex: MythicCodex):
        self.mythic_codex = mythic_codex

    def generate_report(self) -> str:
        """
        Creates a human-readable report of the Mythic Codex contents.
        """
        if not self.mythic_codex.codex:
            logger.info("The Mythic Codex is currently empty.")
            return "Mythic Codex is empty."

        report_lines = ["=== Mythic Codex Report ===\n"]

        for idx, appraised_thread in enumerate(self.mythic_codex.codex, start=1):
            valence_desc = (
                "Positive" if appraised_thread.overall_valence > 0 else
                "Neutral" if appraised_thread.overall_valence == 0 else
                "Negative"
            )
            motifs = ', '.join([m.motif.tag for m in appraised_thread.appraised_motifs])
            report_lines.append(
                f"Thread {idx}: {appraised_thread.thread.motif_name}\n"
                f"    Significance: {appraised_thread.overall_significance:.2f}\n"
                f"    Valence: {valence_desc}\n"
                f"    Urgency: {appraised_thread.overall_urgency:.2f}\n"
                f"    Motifs: {motifs}\n"
            )

        report = '\n'.join(report_lines)
        logger.info("Generated Mythic Codex report.")
        return report

    def print_report(self):
        """
        Prints the Mythic Codex report directly.
        """
        report = self.generate_report()
        print(report)
