from __future__ import annotations

import pytest

from mazerunner_ppo.exact_env import (
    ExactGameConfig,
    _hunger_orb_urgency,
    _milestone_bonus,
)
from mazerunner_ppo.record import summarize_record_runs


def test_record_summary_tracks_tail_and_milestones() -> None:
    metrics = summarize_record_runs(
        [10.0, 60.0, 120.0, 180.0, 240.0, 300.0],
        [0, 1, 2, 3, 4, 5],
        ["hunger", "hunger", "enemy", "hunger", "enemy", "hunger"],
    )
    assert metrics["episodes"] == 6
    assert metrics["median"] == pytest.approx(150.0)
    assert metrics["max"] == pytest.approx(300.0)
    assert metrics["mean_orbs"] == pytest.approx(2.5)
    assert metrics["milestone_rates"]["over_60s"] == pytest.approx(5 / 6)
    assert metrics["milestone_rates"]["over_300s"] == pytest.approx(1 / 6)
    assert metrics["death_counts"] == {"enemy": 2, "hunger": 4}


def test_progressive_hunger_urgency_strengthens_near_zero() -> None:
    assert _hunger_orb_urgency(0.8, 0.75, 2.0) == 0.0
    assert _hunger_orb_urgency(0.7, 0.75, 0.0) == 1.0
    moderate = _hunger_orb_urgency(0.5, 0.75, 2.0)
    urgent = _hunger_orb_urgency(0.1, 0.75, 2.0)
    assert moderate >= 1.0
    assert urgent > moderate


def test_milestone_bonus_only_fires_when_crossed() -> None:
    milestones = ((60.0, 10.0), (120.0, 20.0), (180.0, 30.0))
    assert _milestone_bonus(59.9, 60.1, milestones) == pytest.approx(10.0)
    assert _milestone_bonus(60.1, 119.9, milestones) == 0.0
    assert _milestone_bonus(119.9, 180.1, milestones) == pytest.approx(50.0)


def test_record_config_validates_milestones() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        ExactGameConfig(
            survival_milestone_rewards=((120.0, 1.0), (60.0, 1.0))
        ).validate()
    with pytest.raises(ValueError, match="non-negative"):
        ExactGameConfig(hunger_orb_urgency_power=-1.0).validate()
