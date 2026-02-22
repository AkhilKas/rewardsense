"""
Synthetic user profile generator for RewardSense.

Generates diverse user profiles with assigned spending archetypes,
card portfolios, and redemption preferences. All output is fully
reproducible when using a fixed seed.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from data_pipeline.generators.config import (
    ARCHETYPE_DISTRIBUTION,
    CARD_PORTFOLIO_TEMPLATES,
    DEFAULT_NUM_USERS,
    DEFAULT_SEED,
    REDEMPTION_PREFERENCE_WEIGHTS,
    SPENDING_ARCHETYPES,
)

logger = logging.getLogger(__name__)


class UserProfileGenerator:
    """Generates synthetic user profiles for the RewardSense system.

    Each profile includes a spending archetype, assigned card portfolio,
    monthly budget, and redemption preference. The generator ensures
    deterministic output for a given seed.

    Parameters
    ----------
    num_users : int
        Number of user profiles to generate.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        num_users: int = DEFAULT_NUM_USERS,
        seed: int = DEFAULT_SEED,
    ) -> None:
        self.num_users = num_users
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._archetype_map = {a.name: a for a in SPENDING_ARCHETYPES}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self) -> pd.DataFrame:
        """Generate all user profiles and return as a DataFrame.

        Returns
        -------
        pd.DataFrame
            Columns: user_id, archetype, monthly_budget, cards,
            redemption_preference, age_group, location_type
        """
        logger.info(
            "Generating %d user profiles with seed=%d",
            self.num_users,
            self.seed,
        )

        archetypes = self._assign_archetypes()
        profiles: List[Dict] = []

        for i, arch_name in enumerate(archetypes):
            user_id = f"user_{i + 1:04d}"
            archetype = self._archetype_map[arch_name]
            budget = self._sample_monthly_budget(archetype)
            cards = self._assign_card_portfolio(arch_name)
            redemption_pref = self._sample_redemption_preference()
            age_group = self._sample_age_group(arch_name)
            location_type = self._sample_location_type(arch_name)

            profiles.append(
                {
                    "user_id": user_id,
                    "archetype": arch_name,
                    "monthly_budget": round(budget, 2),
                    "cards": cards,
                    "redemption_preference": redemption_pref,
                    "age_group": age_group,
                    "location_type": location_type,
                }
            )

        df = pd.DataFrame(profiles)
        logger.info(
            "Generated %d profiles across %d archetypes",
            len(df),
            df["archetype"].nunique(),
        )
        return df

    def generate_user_cards_mapping(
        self, profiles_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Explode user profiles into a user_id ↔ card_id mapping table.

        This is the 'Synthetic User's credit cards owned' dataset
        referenced in the design doc.

        Parameters
        ----------
        profiles_df : pd.DataFrame, optional
            Pre-generated profiles. If None, calls ``self.generate()``.

        Returns
        -------
        pd.DataFrame
            Columns: user_id, card_id, redemption_preference
        """
        if profiles_df is None:
            profiles_df = self.generate()

        rows: List[Dict] = []
        for _, row in profiles_df.iterrows():
            for card in row["cards"]:
                rows.append(
                    {
                        "user_id": row["user_id"],
                        "card_id": card,
                        "redemption_preference": row["redemption_preference"],
                    }
                )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assign_archetypes(self) -> List[str]:
        """Assign an archetype to each user based on configured distribution."""
        names = list(ARCHETYPE_DISTRIBUTION.keys())
        weights = np.array([ARCHETYPE_DISTRIBUTION[n] for n in names])
        weights = weights / weights.sum()
        return list(self._rng.choice(names, size=self.num_users, p=weights))

    def _sample_monthly_budget(self, archetype) -> float:
        """Sample a monthly budget uniformly within the archetype range."""
        lo, hi = archetype.monthly_budget_range
        return float(self._rng.uniform(lo, hi))

    def _assign_card_portfolio(self, archetype_name: str) -> List[str]:
        """Select a card portfolio template that aligns with the archetype."""
        affinity = {
            "young_professional": ["starter_cashback", "dining_focused", "all_rounder"],
            "suburban_family": ["grocery_focused", "all_rounder", "starter_cashback"],
            "frequent_traveler": ["travel_focused", "dining_focused", "premium_stack"],
            "budget_conscious": ["starter_cashback", "all_rounder"],
            "high_roller": ["premium_stack", "travel_focused"],
            "minimal_user": ["starter_cashback"],
            "category_specialist": ["grocery_focused", "all_rounder", "dining_focused"],
        }
        candidates = affinity.get(archetype_name, list(CARD_PORTFOLIO_TEMPLATES.keys()))
        template_name = self._rng.choice(candidates)
        return CARD_PORTFOLIO_TEMPLATES[template_name]

    def _sample_redemption_preference(self) -> str:
        """Sample a redemption preference based on global weights."""
        prefs = list(REDEMPTION_PREFERENCE_WEIGHTS.keys())
        weights = np.array([REDEMPTION_PREFERENCE_WEIGHTS[p] for p in prefs])
        weights = weights / weights.sum()
        return str(self._rng.choice(prefs, p=weights))

    def _sample_age_group(self, archetype_name: str) -> str:
        """Sample an age group correlated with archetype."""
        age_affinities: Dict[str, Dict[str, float]] = {
            "young_professional": {"18-25": 0.4, "26-35": 0.5, "36-50": 0.1},
            "suburban_family": {"26-35": 0.3, "36-50": 0.5, "51-65": 0.2},
            "frequent_traveler": {"26-35": 0.3, "36-50": 0.4, "51-65": 0.3},
            "budget_conscious": {
                "18-25": 0.3,
                "26-35": 0.3,
                "36-50": 0.2,
                "51-65": 0.15,
                "65+": 0.05,
            },
            "high_roller": {"36-50": 0.4, "51-65": 0.4, "65+": 0.2},
            "minimal_user": {"18-25": 0.5, "65+": 0.3, "51-65": 0.2},
            "category_specialist": {"26-35": 0.3, "36-50": 0.4, "51-65": 0.3},
        }
        dist = age_affinities.get(archetype_name, {"26-35": 0.5, "36-50": 0.5})
        groups = list(dist.keys())
        weights = np.array(list(dist.values()))
        weights = weights / weights.sum()
        return str(self._rng.choice(groups, p=weights))

    def _sample_location_type(self, archetype_name: str) -> str:
        """Sample urban/suburban/rural location correlated with archetype."""
        loc_affinities: Dict[str, Dict[str, float]] = {
            "young_professional": {"urban": 0.7, "suburban": 0.25, "rural": 0.05},
            "suburban_family": {"urban": 0.1, "suburban": 0.75, "rural": 0.15},
            "frequent_traveler": {"urban": 0.5, "suburban": 0.4, "rural": 0.1},
            "budget_conscious": {"urban": 0.3, "suburban": 0.4, "rural": 0.3},
            "high_roller": {"urban": 0.6, "suburban": 0.35, "rural": 0.05},
            "minimal_user": {"urban": 0.3, "suburban": 0.3, "rural": 0.4},
            "category_specialist": {"urban": 0.3, "suburban": 0.5, "rural": 0.2},
        }
        dist = loc_affinities.get(
            archetype_name, {"urban": 0.4, "suburban": 0.4, "rural": 0.2}
        )
        locs = list(dist.keys())
        weights = np.array(list(dist.values()))
        weights = weights / weights.sum()
        return str(self._rng.choice(locs, p=weights))
