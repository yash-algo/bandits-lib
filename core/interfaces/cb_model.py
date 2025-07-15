# core/interfaces/model.py

from __future__ import annotations
from typing import Protocol, Any, Sequence, Tuple, Optional, Dict


class CBModel(Protocol):
    """
    Contextual‐bandit model interface using tuple‐based rows:
      - `predict`  : one (context, candidates) tuple → (action, score)
      - `batch_predict` : list of (context, candidates) tuples → list of (action, score)
      - `update`   : one (context, action, reward, prob_logged) tuple → None
      - `batch_update` : list of (context, action, reward, prob_logged) tuples → None
      Exploration policy is fixed at init and exposed via a getter.
    """

    def __init__(
        self,
        exploration_strategy: str,
        **strategy_params: Any
    ) -> None:
        """
        :param exploration_strategy: e.g. "epsilon_greedy", "lin_ucb", "thompson"
        :param strategy_params:      hyperparameters (ε, α, λ, priors…)
        """

    def get_exploration_strategy(self) -> Tuple[str, Dict[str, Any]]:
        """Return the current exploration policy name and its parameters."""

    def predict(
        self,
        row: Tuple[Any, Sequence[Any]],
        eval_mode: bool = False
    ) -> Tuple[Any, float]:
        """
        Single‐row inference.

        :param row:      (context, candidate_set)
                         - context: Any (e.g. feature array or dict)
                         - candidate_set: Sequence[Any] (actions or action‐features)
        :param eval_mode: if True, disable exploration (pure exploitation)
        :returns:        (chosen_action, score)
        """

    def batch_predict(
        self,
        rows: Sequence[Tuple[Any, Sequence[Any]]],
        eval_mode: bool = False
    ) -> Sequence[Tuple[Any, float]]:
        """
        Batch inference over N rows.

        :param rows:      list of N tuples (context, candidate_set)
        :param eval_mode: if True, disable exploration
        :returns:        list of N (action, score) pairs
        """

    def update(
        self,
        interaction: Tuple[Any, Any, float, Optional[float]]
    ) -> None:
        """
        Single‐step online update.

        :param interaction: (context, action, reward, prob_logged)
                            - context: Any
                            - action:  Any
                            - reward:  float
                            - prob_logged: Optional[float] (for IPS/DR)
        """

    def batch_update(
        self,
        interactions: Sequence[Tuple[Any, Any, float, Optional[float]]]
    ) -> None:
        """
        Batch‐style update.

        :param interactions: list of tuples
                             (context, action, reward, prob_logged)
        """

    def reset(self) -> None:
        """Reinitialize internal state for fresh experiments."""

    def save(
        self,
        name: str,
        version: str,
        registry_root: str,
        artifact_subpath: str = "models"
    ) -> str:
        """
        Persist model state via ModelStore (fs/redis/s3).

        :returns: URI or path of the saved artifact.
        """

    @classmethod
    def load(
        cls,
        name: str,
        version: str,
        registry_root: str,
        artifact_subpath: str = "models"
    ) -> CBModel:
        """Reconstruct a `CBModel` from its saved artifact."""
