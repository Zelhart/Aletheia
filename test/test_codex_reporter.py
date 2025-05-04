import unittest
from core.mythos import MythosForge
from core.appraisal import AppraisalEngine
from core.affect import AffectiveModulation
from memory.mythic_codex import MythicCodex
from tools.codex_reporter import CodexReporter

class TestCodexReporter(unittest.TestCase):

    def setUp(self):
        # Set up minimal environment for Mythic Codex
        self.mythos_forge = MythosForge()
        self.mythos_forge.create_motif("Test Motif A", 0.7)
        self.mythos_forge.create_motif("Test Motif B", 0.5)
        self.mythos_forge.create_thread("Test Thread", ["Test Motif A", "Test Motif B"])

        self.appraisal_engine = AppraisalEngine(self.mythos_forge)
        self.affective_modulation = AffectiveModulation(self.appraisal_engine)

        # Appraise threads to populate appraised threads
        self.appraisal_engine.appraise_threads()

        # Create Mythic Codex and crystallize current threads
        self.mythic_codex = MythicCodex(self.appraisal_engine)
        self.mythic_codex.crystallize()

        self.reporter = CodexReporter(self.mythic_codex)

    def test_generate_report(self):
        report = self.reporter.generate_report()
        self.assertIn("Test Thread", report)
        self.assertIn("Test Motif A", report)
        self.assertIn("Test Motif B", report)
        print("\nGenerated Report:\n", report)

    def test_empty_codex_report(self):
        empty_codex = MythicCodex(self.appraisal_engine)
        empty_reporter = CodexReporter(empty_codex)
        report = empty_reporter.generate_report()
        self.assertEqual(report, "Mythic Codex is empty.")

if __name__ == "__main__":
    unittest.main()
