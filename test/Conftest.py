import pytest

@pytest.fixture(scope="session")
def setup_cognitive_core():
    from core.cognitive_core import CognitiveCore
    core = CognitiveCore()
    core.appraisal_engine.initialize_motifs_and_threads()  # if such method exists
    return core
