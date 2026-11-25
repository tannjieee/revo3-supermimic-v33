"""Relative-improvement patience for a metric that should decrease."""

from dataclasses import dataclass
import math


@dataclass
class EarlyStopping:
    patience: int = 0
    min_relative_improvement: float = 0.01
    reference: float = float("inf")
    bad_epochs: int = 0

    def update(self, value):
        if not math.isfinite(value):
            raise ValueError("Early-stopping metric must be finite")
        if value < self.reference * (1 - self.min_relative_improvement):
            self.reference = value
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return self.patience > 0 and self.bad_epochs >= self.patience
