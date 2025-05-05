#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for Reflex System and Actuator Bridge (Aletheia)
=======================================================
"""

import pytest
from core.affect import AffectiveModulation
from core.reflex_system import ReflexSystem
from core.actuator_bridge import ActuatorBridge
from core.action import ActionProposal

def test_positive_reflex_trigger():
    affect_mod = AffectiveModulation()
    affect_mod.current_affect.valence = 0.8
    reflex = ReflexSystem(affect_mod)
    action = reflex.check_reflexes()
    assert action is not None
    assert action.intent.motif == "celebrate"

def test_negative_reflex_trigger():
    affect_mod = AffectiveModulation()
    affect_mod.current_affect.valence = -0.6
    reflex = ReflexSystem(affect_mod)
    action = reflex.check_reflexes()
    assert action is not None
    assert action.intent.motif == "withdraw"

def test_actuator_execution():
    bridge = ActuatorBridge()
    dummy_action = ActionProposal(
        description="Test action",
        intent=None,  # Normally an Intent object
        urgency=0.3
    )
    dummy_action.intent = type("Intent", (), {"motif": "test_motif"})()
    result = bridge.execute(dummy_action)
    assert result["success"] == True
