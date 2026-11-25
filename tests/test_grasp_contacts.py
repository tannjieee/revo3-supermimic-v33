import numpy as np
import pytest
from viserdex.supermimic.grasp_contacts import accumulate_object_contacts


@pytest.mark.parametrize('world_count', [1, 4, 256])
def test_reduction_matches_independent_per_world_contact_oracle(world_count):
    rng = np.random.default_rng(410)
    count = np.zeros(world_count, dtype=int)
    depth = np.zeros(world_count)
    expected_count = count.copy()
    expected_depth = depth.copy()
    for n in (0, 17, 3000, 10):
        geoms = rng.integers(0, 20, (n, 2))
        worlds = rng.integers(-1, world_count+1, n)
        distances = rng.uniform(-.004, .003, n).astype(np.float32)
        active = rng.random(n) < .75
        objects = [7, 13]
        for env in range(world_count):
            relevant = [float(distances[i]) for i in range(n)
                        if active[i] and worlds[i] == env and any(g in objects for g in geoms[i])]
            if relevant:
                expected_count[env] += 1
                expected_depth[env] = max(expected_depth[env], max(0., -min(relevant)))
        accumulate_object_contacts(geoms, worlds, distances, active, objects, count, depth)
        np.testing.assert_array_equal(count, expected_count)
        np.testing.assert_array_equal(depth, expected_depth)
