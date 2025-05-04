import json
import csv
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class LogManager:
    """
    Handles persistent logging of Codex Crystals and Mythopoetic Narratives
    for long-term analysis and continuity.
    """

    def __init__(self, log_dir="logs"):
        self.log_dir = Path(log_dir)
        self.crystals_path = self.log_dir / "codex_crystals.jsonl"
        self.narratives_path = self.log_dir / "narratives.jsonl"

        self.log_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"LogManager initialized. Logs directory: {self.log_dir.resolve()}")

    def log_crystal(self, crystal):
        """
        Save a Codex Crystal to disk.
        """
        record = {
            "timestep": crystal.timestep,
            "dominant_emotion": crystal.dominant_emotion,
            "prevailing_motifs": crystal.prevailing_motifs,
            "emotional_signature": crystal.emotional_signature,
            "intents_summary": crystal.intents_summary
        }
        with self.crystals_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        logger.info(f"Codex Crystal for timestep {crystal.timestep} logged.")

    def log_narrative(self, narrative, timestep):
        """
        Save a Mythopoetic Narrative to disk.
        """
        record = {
            "timestep": timestep,
            "title": narrative.title,
            "reflection": narrative.reflection,
            "future_pull": narrative.future_pull
        }
        with self.narratives_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        logger.info(f"Narrative for timestep {timestep} logged.")

    def export_crystals_csv(self, csv_path="logs/codex_crystals.csv"):
        """
        Export Codex Crystals to CSV for easy review.
        """
        csv_file = Path(csv_path)
        with csv_file.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestep", "dominant_emotion", "prevailing_motifs",
                             "emotional_signature", "intents_summary"])
            with self.crystals_path.open("r", encoding="utf-8") as jf:
                for line in jf:
                    rec = json.loads(line)
                    writer.writerow([
                        rec["timestep"],
                        rec["dominant_emotion"],
                        ", ".join(rec["prevailing_motifs"]),
                        str(rec["emotional_signature"]),
                        ", ".join(rec["intents_summary"])
                    ])
        logger.info(f"Codex Crystals exported to {csv_file.resolve()}.")

    def export_narratives_csv(self, csv_path="logs/narratives.csv"):
        """
        Export Narratives to CSV for easy review.
        """
        csv_file = Path(csv_path)
        with csv_file.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestep", "title", "reflection", "future_pull"])
            with self.narratives_path.open("r", encoding="utf-8") as jf:
                for line in jf:
                    rec = json.loads(line)
                    writer.writerow([
                        rec["timestep"],
                        rec["title"],
                        rec["reflection"],
                        rec["future_pull"]
                    ])
        logger.info(f"Narratives exported to {csv_file.resolve()}.")
