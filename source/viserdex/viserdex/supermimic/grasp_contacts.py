"""Exact per-world reduction of active object contacts for grasp certification."""
import numpy as np


def accumulate_object_contacts(geoms, worlds, distances, active, object_geoms, samples, penetration):
    mask = np.isin(geoms, object_geoms).any(axis=1) & active
    # Ignore out-of-world storage slots as the per-environment reference loop did.
    mask &= (worlds >= 0) & (worlds < len(samples))
    ids = worlds[mask]
    if len(ids):
        samples += np.bincount(ids, minlength=len(samples)) > 0
        np.maximum.at(penetration, ids, np.maximum(0., -distances[mask]))
