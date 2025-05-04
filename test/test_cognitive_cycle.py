def test_cognitive_cycle_runs(setup_cognitive_core):
    core = setup_cognitive_core
    result = core.cognitive_cycle()
    assert result is not None
    assert "timestep" in result
    assert "narrative" in result
