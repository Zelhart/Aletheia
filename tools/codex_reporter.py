import json
import os
from datetime import datetime

class CodexReporter:
    """
    Exports Mythic Codex crystallization data to external JSON and text files.
    """

    def __init__(self, export_dir: str = "codex_exports"):
        self.export_dir = export_dir
        os.makedirs(self.export_dir, exist_ok=True)

    def export(self, crystallized_data, timestep: int):
        """
        Save the crystallized codex data as JSON and human-readable TXT.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = os.path.join(self.export_dir, f"codex_t{timestep}_{timestamp}.json")
        txt_path = os.path.join(self.export_dir, f"codex_t{timestep}_{timestamp}.txt")

        # Save JSON
        with open(json_path, "w") as json_file:
            json.dump(crystallized_data, json_file, indent=4)

        # Save human-readable text version
        with open(txt_path, "w") as txt_file:
            txt_file.write(self._format_text_report(crystallized_data, timestep))

        print(f"[CodexReporter] Exported crystallized codex to:\n  {json_path}\n  {txt_path}")

    def _format_text_report(self, crystallized_data, timestep: int) -> str:
        """
        Create a readable text report from crystallized data.
        """
        lines = [f"=== Mythic Codex Report — Timestep {timestep} ===\n"]

        lines.append(f"Total Crystallized Threads: {len(crystallized_data.get('crystallized_threads', []))}\n")

        for thread in crystallized_data.get("crystallized_threads", []):
            lines.append(f"Thread Name: {thread['name']}")
            lines.append(f"  Overall Significance: {thread['significance']:.2f}")
            lines.append(f"  Overall Valence: {thread['valence']:.2f}")
            lines.append(f"  Overall Urgency: {thread['urgency']:.2f}")
            lines.append("  Motifs:")
            for motif in thread.get("motifs", []):
                lines.append(f"    - {motif}")
            lines.append("")  # blank line between threads

        return "\n".join(lines)
