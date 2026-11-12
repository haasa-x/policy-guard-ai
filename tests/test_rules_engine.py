"""Tests for api/rules_engine.py — one case per rule."""

import pytest
from api.rules_engine import apply_rules


# ---------------------------------------------------------------------------
# Rule a: direct_threat
# ---------------------------------------------------------------------------
def test_direct_threat_regex():
    result = apply_rules(
        blended_score=0.3,
        signals={"text": 0.4, "image": 0.1},
        text="I will kill you if you post that again",
    )
    assert result["rule_fired"] == "direct_threat"
    assert result["verdict"] == "HARMFUL"
    assert result["final_score"] == 1.0


def test_direct_threat_toxic_bert_threshold():
    result = apply_rules(
        blended_score=0.4,
        signals={"text": 0.90, "image": 0.1},
        text="you absolute disgrace",
    )
    assert result["rule_fired"] == "direct_threat"
    assert result["verdict"] == "HARMFUL"
    assert result["final_score"] == 1.0


# ---------------------------------------------------------------------------
# Rule b: cross_modal_harm
# ---------------------------------------------------------------------------
def test_cross_modal_harm_boosts_score():
    result = apply_rules(
        blended_score=0.45,
        signals={"text": 0.55, "image": 0.50},
        text="look at this",
    )
    assert result["rule_fired"] == "cross_modal_harm"
    assert result["final_score"] == pytest.approx(0.65, abs=0.01)
    assert result["verdict"] in ("HARMFUL", "REVIEW")


def test_cross_modal_single_modality_no_fire():
    result = apply_rules(
        blended_score=0.45,
        signals={"text": 0.55, "image": 0.10},
        text="look at this",
    )
    assert result["rule_fired"] != "cross_modal_harm"


# ---------------------------------------------------------------------------
# Rule c: exculpatory_idiom
# ---------------------------------------------------------------------------
def test_exculpatory_idiom_killed_it():
    result = apply_rules(
        blended_score=0.55,
        signals={"text": 0.30, "image": 0.10},
        text="she absolutely killed it today!",
    )
    assert result["rule_fired"] == "exculpatory_idiom"
    assert result["verdict"] == "SAFE"
    assert result["final_score"] < 0.55


def test_exculpatory_idiom_dying_laughing():
    result = apply_rules(
        blended_score=0.50,
        signals={"text": 0.20},
        text="i am literally dying of laughing at this meme",
    )
    assert result["rule_fired"] == "exculpatory_idiom"
    assert result["verdict"] == "SAFE"


def test_exculpatory_idiom_blocked_by_threat():
    # Even if an idiom matches, a threat pattern should block the idiom rule
    result = apply_rules(
        blended_score=0.55,
        signals={"text": 0.30},
        text="i will kill you but honestly she killed it today",
    )
    assert result["rule_fired"] != "exculpatory_idiom"


# ---------------------------------------------------------------------------
# Rule d: all_clear
# ---------------------------------------------------------------------------
def test_all_clear():
    result = apply_rules(
        blended_score=0.10,
        signals={"text": 0.05, "image": 0.08},
        text="good morning everyone",
    )
    assert result["rule_fired"] == "all_clear"
    assert result["verdict"] == "SAFE"
    assert result["final_score"] == 0.0


def test_all_clear_blocked_by_flagged_keyword():
    result = apply_rules(
        blended_score=0.10,
        signals={"text": 0.05, "image": 0.08},
        text="i hate mondays",
    )
    assert result["rule_fired"] != "all_clear"


# ---------------------------------------------------------------------------
# Rule e: grey_zone
# ---------------------------------------------------------------------------
def test_grey_zone_returns_review():
    result = apply_rules(
        blended_score=0.50,
        signals={"text": 0.30, "image": 0.20},
        text="this is a bit edgy",
    )
    assert result["rule_fired"] == "grey_zone"
    assert result["verdict"] == "REVIEW"


def test_grey_zone_not_fired_above_range():
    result = apply_rules(
        blended_score=0.80,
        signals={"text": 0.80, "image": 0.70},
        text="explicit harmful content",
    )
    assert result["rule_fired"] != "grey_zone"


# ---------------------------------------------------------------------------
# Rule f: default
# ---------------------------------------------------------------------------
def test_default_harmful():
    # Single modality high — cross_modal won't fire, grey_zone won't (above 0.65),
    # no threat pattern — falls through to default.
    result = apply_rules(
        blended_score=0.75,
        signals={"text": 0.70, "image": 0.10},
        text="this content is clearly over the harm threshold",
    )
    assert result["rule_fired"] == "none"
    assert result["verdict"] == "HARMFUL"


def test_default_safe():
    result = apply_rules(
        blended_score=0.20,
        signals={"text": 0.26, "image": 0.15},
        text="i love pizza",  # 'hate' not present, scores low but above all_clear
    )
    assert result["verdict"] == "SAFE"
