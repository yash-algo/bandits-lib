# core/interfaces/action.py

from __future__ import annotations
from typing import Protocol, Any, Dict


class Action(Protocol):
    """
    Minimal action abstraction for contextual bandits:

      1. get_id        → track which arm
      2. features      → model-input features (vector, namespace map, etc.)
      3. record_stats  → accumulate any outcome stats (reward, counts, ...)
    """

    def get_id(self) -> Any:
        """
        Return a unique identifier for this action (index, name, etc.).
        """

    def features(self) -> Any:
        """
        Return the features used by the model.
        Can be a numeric vector (e.g. np.ndarray) or
        a namespace→dict mapping for VW-ADF.
        """

    def record_stats(self, reward: float, **info: Any) -> None:
        """
        Store outcome statistics for this action.
        
        :param reward: observed reward
        :param info:   optional extra data (e.g. context, logged probability)
        """
