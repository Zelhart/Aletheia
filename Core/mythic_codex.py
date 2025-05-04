import logging
from typing import List
from core.appraisal import AppraisedThread

logger = logging.getLogger(__name__)

class MythicCodex:
    """
    Archives crystallized Mythic Threads into a persistent symbolic memory system.
    
    This enables long-term retention of emotionally and cognitively significant patterns.
    """
    def __init__(self):
        self.crystallized_entries: List[dict] = []

    def archive_thread(self, appraised_thread: AppraisedThread):
        """
        Archives a crystallized thread into the Codex if not already stored.
        """
        entry = {
            "thread_name": appraised_thread.thread.motif_name,
            "significance": appraised_thread.overall_significance,
            "valence": appraised_thread.overall_valence,
            "urgency": appraised_thread.overall_urgency,
            "motifs": [m.motif.tag for m in appraised_thread.appraised_motifs]
        }
        self.crystallized_entries.append(entry)
        logger.info(f"*** Mythic Codex archived thread '{entry['thread_name']}' ***")

    def list_codex(self) -> List[dict]:
        """
        Returns a summary list of crystallized threads.
        """
        return [{
            "thread": entry["thread_name"],
            "significance": entry["significance"],
            "motifs": entry["motifs"]
        } for entry in self.crystallized_entries]

    def describe_codex(self):
        """
        Prints a detailed description of all archived threads to the logger.
        """
        logger.info("=== Mythic Codex Entries ===")
        if not self.crystallized_entries:
            logger.info("No crystallized threads archived yet.")
            return
        for entry in self.crystallized_entries:
            logger.info(
                f"Thread '{entry['thread_name']}' | Significance: {entry['significance']:.2f} | "
                f"Motifs: {', '.join(entry['motifs'])}"
            )
