import numpy as np
import pytest

from horsefold import Structure, basran2022


@pytest.mark.parametrize("missing_on", ["bilateral", "single"])
def test_basran2022_matches_paper_counts(missing_on):
    s = basran2022(missing_on)
    assert s.n_units == 129
    assert s.n_groups == 50
    assert int(s.y.sum()) == 60
    assert (s.counts > 0).all()
    assert s.n_clusters == 65  # 30 case limbs + 35 control limbs
    control_counts = sorted(s.counts[30:].tolist(), reverse=True)
    if missing_on == "bilateral":
        assert control_counts == [4] * 14 + [3] + [2] * 5
    else:
        assert control_counts == [4] * 15 + [2] * 4 + [1]


def test_labels_are_constant_within_group():
    s = basran2022()
    for g in range(s.n_groups):
        assert len(np.unique(s.y[s.group == g])) == 1


def test_from_groups_numbers_positions_and_clusters():
    s = Structure.from_groups([(1, [2]), (0, [1, 3])])
    assert s.group.tolist() == [0, 0, 1, 1, 1, 1]
    assert s.cluster.tolist() == [0, 0, 1, 2, 2, 2]
    assert s.position.tolist() == [0, 1, 0, 0, 1, 2]
    assert s.y.tolist() == [1, 1, 0, 0, 0, 0]


@pytest.mark.parametrize("groups", [[(2, [1])], [(1, [])], [(0, [0])]])
def test_from_groups_rejects_bad_input(groups):
    with pytest.raises(ValueError):
        Structure.from_groups(groups)
