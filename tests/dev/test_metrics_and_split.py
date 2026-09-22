from __future__ import annotations

import numpy as np

from orbit_predict.metrics import competition_huber_score, metric_dict, rmse
from orbit_predict.split import make_split


def test_BR01_Huber점수_대회공식과스케일적용():
    actual = np.array([0.0, 10.0])
    predicted = np.array([1.0, 0.0])
    assert competition_huber_score(actual, predicted) == 19_000.0
    assert rmse(actual, predicted) == np.sqrt(50.5)
    assert metric_dict(actual, predicted)["max_absolute_error"] == 10.0


def test_FR02_고정분할_시드와ceil_비교구간사용():
    split = make_split(11, seed=42, comparison_fraction=0.2)
    expected_permutation = np.random.default_rng(42).permutation(11)
    assert len(split.comparison_indices) == 3
    assert split.comparison_indices.tolist() == expected_permutation[:3].tolist()
    assert split.development_indices.tolist() == expected_permutation[3:].tolist()
    assert set(split.development_indices).isdisjoint(set(split.comparison_indices))


def test_NFR01_분할재현성_동일시드동일인덱스():
    first = make_split(100)
    second = make_split(100)
    assert np.array_equal(first.development_indices, second.development_indices)
    assert np.array_equal(first.comparison_indices, second.comparison_indices)


def test_INT_빈측정배열_명시적오류발생():
    try:
        rmse([], [])
    except ValueError as error:
        assert "at least one" in str(error)
    else:
        raise AssertionError("empty metric input must be rejected")
