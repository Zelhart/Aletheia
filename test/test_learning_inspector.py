import pytest
from tools.learning_inspector import LearningInspector

def test_learning_summary(mock_cognitive_core):
    # Run a few cycles to accumulate learning data
    for _ in range(5):
        mock_cognitive_core.cognitive_cycle()

    inspector = LearningInspector(mock_cognitive_core.echo_meridian)
    summary = inspector.summarize_motif_performance()

    assert isinstance(summary, dict)
    assert len(summary) > 0

def test_latest_learning_reflection(mock_cognitive_core):
    for _ in range(3):
        mock_cognitive_core.cognitive_cycle()

    inspector = LearningInspector(mock_cognitive_core.echo_meridian)
    reflection = inspector.latest_learning_reflection()

    assert isinstance(reflection, str)
    assert "outcome" in reflection
