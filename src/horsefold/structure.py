"""Three-level sample layout: groups (e.g. horses) contain clusters (limbs), which contain units (bones)."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, eq=False)
class Structure:
    """Index arrays describing which group and cluster every unit belongs to.

    All arrays have one entry per unit. Cluster indices are unique across groups.
    ``names`` is used only for labels in figures and reports.
    """

    group: np.ndarray
    cluster: np.ndarray
    position: np.ndarray
    y: np.ndarray
    names: tuple[str, str, str] = ("group", "cluster", "unit")

    @classmethod
    def from_groups(
        cls,
        groups: Sequence[tuple[int, Sequence[int]]],
        names: tuple[str, str, str] = ("group", "cluster", "unit"),
    ) -> "Structure":
        """Build from ``[(label, [units in cluster 1, units in cluster 2, ...]), ...]``, one tuple per group.

        Every unit inherits its group's label (0 or 1). ``position`` numbers the units inside a cluster,
        e.g. 0 = medial and 1 = lateral bone of the same limb.
        """
        group, cluster, position, y = [], [], [], []
        c = 0
        for g, (label, clusters) in enumerate(groups):
            if label not in (0, 1):
                raise ValueError(f"group {g}: label must be 0 or 1, got {label!r}")
            if not clusters or min(clusters) < 1:
                raise ValueError(f"group {g}: every group needs at least one cluster with at least one unit")
            for n_units in clusters:
                for b in range(n_units):
                    group.append(g)
                    cluster.append(c)
                    position.append(b)
                    y.append(label)
                c += 1
        return cls(np.array(group), np.array(cluster), np.array(position), np.array(y, float), names)

    @property
    def n_units(self) -> int:
        return len(self.group)

    @property
    def n_groups(self) -> int:
        return int(self.group.max()) + 1

    @property
    def n_clusters(self) -> int:
        return int(self.cluster.max()) + 1

    @property
    def counts(self) -> np.ndarray:
        """Number of units in each group."""
        return np.bincount(self.group, minlength=self.n_groups)

    def summary(self) -> dict:
        return {
            "names": list(self.names),
            "units": self.n_units,
            "groups": self.n_groups,
            "clusters": self.n_clusters,
            "positive_units": int(self.y.sum()),
            "negative_units": int(self.n_units - self.y.sum()),
            "units_per_group": np.bincount(self.counts).tolist(),
        }


def basran2022(missing_on: str = "bilateral") -> Structure:
    """Sample layout of Basran et al. (2022), Animals 12(21):3033, p.3 and Figure 1.

    Cases: 30 horses, one limb each, medial and lateral proximal sesamoid bone (60 bones).
    Controls: 20 horses with 35 limbs (15 with both forelimbs, 5 with one) and 69 bones, because one
    lateral bone image was corrupted. The paper does not say which control horse lost that bone:
    ``missing_on="bilateral"`` puts it on a two-limb horse, ``"single"`` on a one-limb horse.
    Totals: 129 bones, 65 limbs, 50 horses.
    """
    if missing_on not in ("bilateral", "single"):
        raise ValueError("missing_on must be 'bilateral' or 'single'")
    cases = [(1, [2]) for _ in range(30)]
    controls = [(0, [2, 2]) for _ in range(15)] + [(0, [2]) for _ in range(5)]
    if missing_on == "bilateral":
        controls[0] = (0, [2, 1])
    else:
        controls[-1] = (0, [1])
    return Structure.from_groups(cases + controls, names=("horse", "limb", "bone"))
