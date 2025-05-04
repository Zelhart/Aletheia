def test_crystal_logging(setup_cognitive_core):
    core = setup_cognitive_core
    result = core.cognitive_cycle()
    assert result["narrative"] is not None

    # Check that logs exist
    import os
    assert os.path.exists("logs/codex_crystals.jsonl")
    assert os.path.exists("logs/narratives.jsonl")
