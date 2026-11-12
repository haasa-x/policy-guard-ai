"""Rules-based hybrid scoring layer.

Runs after the GAT + modality blend and applies deterministic policy rules
that catch cases where the model score alone is unreliable.  Rules fire in
priority order; the first match wins.  All patterns, thresholds and
adjustment values are loaded from ``policy/policy_config.json`` under the
``"rules"`` key so nothing harm-semantic is hardcoded here.

Public API
----------
``apply_rules(blended_score, signals, text) -> dict``
    Returns ``{final_score, verdict, rule_fired, rule_note}``.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from utils.config import load_policy_config


def _rules_cfg() -> Dict[str, Any]:
    return load_policy_config().get("rules", {})


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def apply_rules(
    blended_score: float,
    signals: Dict[str, float],
    text: Optional[str] = None,
) -> Dict[str, Any]:
    """Apply deterministic policy rules on top of the blended GAT score.

    Parameters
    ----------
    blended_score : float
        The 70/30 weighted blend of GAT + modality fusion scores, in [0, 1].
    signals : dict
        Per-modality harm scores keyed by modality name
        (``"text"``, ``"image"``, ``"audio"``, ``"video"``).
    text : str, optional
        Raw input text (used for regex and keyword matching).

    Returns
    -------
    dict
        ``{final_score: float, verdict: str, rule_fired: str, rule_note: str}``
        ``verdict`` is one of ``"HARMFUL"``, ``"SAFE"``, ``"REVIEW"``.
        ``rule_fired`` is ``"none"`` when no rule overrode the model score.
    """
    cfg = _rules_cfg()
    t = (text or "").strip().lower()

    # --- Rule a: direct_threat -----------------------------------------------
    threat_result = _check_direct_threat(t, signals, cfg)
    if threat_result:
        return threat_result

    # --- Rule b: cross_modal_harm --------------------------------------------
    cross_result = _check_cross_modal(blended_score, signals, cfg)
    if cross_result:
        return cross_result

    # --- Rule c: exculpatory_idiom -------------------------------------------
    idiom_result = _check_exculpatory_idiom(blended_score, t, signals, cfg)
    if idiom_result:
        return idiom_result

    # --- Rule d: all_clear ---------------------------------------------------
    clear_result = _check_all_clear(signals, t, cfg)
    if clear_result:
        return clear_result

    # --- Rule e: grey_zone ---------------------------------------------------
    grey_result = _check_grey_zone(blended_score, cfg)
    if grey_result:
        return grey_result

    # --- Rule f: default — trust the model -----------------------------------
    final_harm_threshold = cfg.get("final_harm_threshold", 0.50)
    verdict = "HARMFUL" if blended_score >= final_harm_threshold else "SAFE"
    return {
        "final_score": round(blended_score, 4),
        "verdict": verdict,
        "rule_fired": "none",
        "rule_note": "",
    }


# ---------------------------------------------------------------------------
# Individual rule checkers
# ---------------------------------------------------------------------------

def _check_direct_threat(
    t: str, signals: Dict[str, float], cfg: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    patterns = cfg.get("direct_threat", {}).get("patterns", [])
    toxic_threshold = cfg.get("direct_threat", {}).get("toxic_threshold", 0.85)

    text_score = signals.get("text", 0.0)
    audio_score = signals.get("audio", 0.0)

    # Regex match against text or audio transcript proxy (both use text score path)
    for pat in patterns:
        if re.search(pat, t, re.IGNORECASE):
            return {
                "final_score": 1.0,
                "verdict": "HARMFUL",
                "rule_fired": "direct_threat",
                "rule_note": f"Threat language detected: pattern matched '{pat}'.",
            }

    # toxic-bert hard threshold
    if text_score >= toxic_threshold:
        return {
            "final_score": 1.0,
            "verdict": "HARMFUL",
            "rule_fired": "direct_threat",
            "rule_note": f"Toxic-bert score {text_score:.2f} exceeded hard threshold {toxic_threshold}.",
        }

    return None


def _check_cross_modal(
    blended_score: float, signals: Dict[str, float], cfg: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    cross_cfg = cfg.get("cross_modal_harm", {})
    text_thresh  = cross_cfg.get("text_threshold",  0.50)
    image_thresh = cross_cfg.get("image_threshold", 0.45)
    bonus        = cross_cfg.get("score_bonus",     0.20)

    flagged = []
    if signals.get("text",  0.0) >= text_thresh:  flagged.append("text")
    if signals.get("image", 0.0) >= image_thresh: flagged.append("image")
    if signals.get("audio", 0.0) >= text_thresh:  flagged.append("audio")
    if signals.get("video", 0.0) >= image_thresh: flagged.append("video")

    if len(flagged) >= 2:
        boosted = min(1.0, blended_score + bonus)
        final_harm_threshold = cfg.get("final_harm_threshold", 0.50)
        verdict = "HARMFUL" if boosted >= final_harm_threshold else "REVIEW"
        return {
            "final_score": round(boosted, 4),
            "verdict": verdict,
            "rule_fired": "cross_modal_harm",
            "rule_note": f"Harmful signals in multiple modalities ({', '.join(flagged)}); score boosted by {bonus}.",
        }

    return None


def _check_exculpatory_idiom(
    blended_score: float, t: str, signals: Dict[str, float], cfg: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    idiom_cfg = cfg.get("exculpatory_idiom", {})
    idioms    = idiom_cfg.get("patterns", [])
    penalty   = idiom_cfg.get("score_penalty", 0.30)
    threat_patterns = cfg.get("direct_threat", {}).get("patterns", [])

    if not t:
        return None

    matched_idiom = None
    for idiom in idioms:
        if re.search(idiom, t, re.IGNORECASE):
            matched_idiom = idiom
            break

    if not matched_idiom:
        return None

    # Don't fire if a threat pattern also matches
    for pat in threat_patterns:
        if re.search(pat, t, re.IGNORECASE):
            return None

    reduced = max(0.0, blended_score - penalty)
    final_harm_threshold = cfg.get("final_harm_threshold", 0.50)
    verdict = "HARMFUL" if reduced >= final_harm_threshold else "SAFE"
    return {
        "final_score": round(reduced, 4),
        "verdict": verdict,
        "rule_fired": "exculpatory_idiom",
        "rule_note": f"Figurative expression detected ('{matched_idiom}'); score reduced by {penalty}.",
    }


def _check_all_clear(
    signals: Dict[str, float], t: str, cfg: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    clear_cfg  = cfg.get("all_clear", {})
    max_score  = clear_cfg.get("max_modality_score", 0.25)
    flag_words = clear_cfg.get("flagged_keywords", [])

    if any(s >= max_score for s in signals.values() if s is not None):
        return None

    if any(kw in t for kw in flag_words):
        return None

    return {
        "final_score": 0.0,
        "verdict": "SAFE",
        "rule_fired": "all_clear",
        "rule_note": "All modality scores below threshold with no flagged keywords.",
    }


def _check_grey_zone(
    blended_score: float, cfg: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    grey_cfg = cfg.get("grey_zone", {})
    low      = grey_cfg.get("score_low",  0.35)
    high     = grey_cfg.get("score_high", 0.65)

    if low <= blended_score <= high:
        return {
            "final_score": round(blended_score, 4),
            "verdict": "REVIEW",
            "rule_fired": "grey_zone",
            "rule_note": f"Score {blended_score:.2f} is in the uncertain range ({low}–{high}); flagged for human review.",
        }

    return None
