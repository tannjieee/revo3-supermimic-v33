import pytest
from viserdex.utils.early_stopping import EarlyStopping


def test_patience_requires_five_epochs_without_significant_improvement():
    rule = EarlyStopping(5, 0.01)
    assert not rule.update(1.4)
    for value in (1.395, 1.5, 1.39, 1.4):
        assert not rule.update(value)
    assert rule.update(1.39)


def test_small_improvements_accumulate_against_last_significant_best():
    rule = EarlyStopping(5, 0.01)
    assert not rule.update(1.0)
    assert not rule.update(0.995)
    assert rule.bad_epochs == 1
    assert not rule.update(0.985)
    assert rule.bad_epochs == 0
    assert rule.reference == 0.985


def test_disabled_rule_never_stops_and_rejects_nan():
    rule = EarlyStopping(0)
    for _ in range(10):
        assert not rule.update(1.0)
    with pytest.raises(ValueError):
        rule.update(float("nan"))
