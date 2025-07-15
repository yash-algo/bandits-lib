# core/actions/adf_action.py

from typing import Any, Dict, List, Optional
from core.interfaces.action import Action


class ADFAction(Action):
    """
    Action with VW-ADF (action-dependent) features.

    - get_id()      → returns the action’s unique identifier
    - features()    → returns the namespace→feature dict for VW ADF
    - record_stats  → collects observed rewards and optional metadata
    """

    def __init__(
        self,
        namespace_features: Dict[str, Dict[str, Any]],
        action_id: Optional[Any] = None
    ) -> None:
        """
        :param namespace_features:
            e.g. {"A": {"eventType": "atc"},
                  "B": {"timeWindow": 480},
                  "D": {"threshold_low": 10, "threshold_high": 20}}
        :param action_id:
            Optional unique identifier (index, name, etc.)
        """
        self._features = namespace_features
        self._id = action_id
        self._stats: List[Dict[str, Any]] = []

    def get_id(self) -> Any:
        """Unique identifier for this action."""
        return self._id

    def features(self) -> Dict[str, Dict[str, Any]]:
        """
        Namespace→feature mapping for VW-ADF.
        
        VW adapter will render this as:
          "|<ns> key1=val1 key2=val2  ..."
        """
        return self._features

    def record_stats(self, reward: float, **info: Any) -> None:
        """
        Record an observed outcome for this action.

        :param reward: observed reward
        :param info:   any additional data (e.g. context, logged prob)
        """
        entry = {"reward": reward, **info}
        self._stats.append(entry)

    @property
    def stats(self) -> List[Dict[str, Any]]:
        """
        Access collected stats for logging or analysis.
        """
        return self._stats

    def clear_stats(self) -> None:
        """
        Reset the recorded stats.
        """
        self._stats.clear()
