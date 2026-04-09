"""Persona-aware post-processing of scored card rankings.

The PersonaModifier runs *after* the base PersonalizedScorer so it never
touches the ML pipeline — it is a pure application-layer adjustment.

Adjustment logic
----------------
For each card in the ranked list:

1. Category boost  — reward_amount is multiplied by the merged boost for the
   transaction category.  The merged value is the average of all active
   personas' boosts for that category (defaulting to 1.0 when a persona has
   no entry for it).

2. Extra fee penalty — each persona's annual_fee_multiplier shifts the
   monthly-amortised fee cost up or down.  The delta is:

       extra_penalty = (merged_multiplier - 1.0) * annual_fee / 12

   A multiplier of 2.0 adds an extra month's worth of fee as a penalty;
   0.5 credits back half a month's fee.

The final adjusted score is:
    adjusted = reward_amount * category_boost - extra_fee_penalty

Cards are then re-sorted descending and ranks reassigned from 1.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import yaml


_DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "config", "personas.yaml"
)


class PersonaModifier:
    """Load persona definitions and apply them to a scored card list."""

    def __init__(self, config_path: str = _DEFAULT_CONFIG_PATH) -> None:
        path = os.path.normpath(config_path)
        with open(path) as fh:
            raw = yaml.safe_load(fh)
        self._personas: Dict[str, Dict[str, Any]] = raw.get("personas", {})

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply(
        self,
        ranked: List[Dict[str, Any]],
        active_personas: List[str],
        category: str,
    ) -> Dict[str, Any]:
        """Return adjusted ranked list and persona context metadata.

        Parameters
        ----------
        ranked:
            Output of ``PersonalizedScorer.score()["ranked"]``.
        active_personas:
            Keys from ``personas.yaml`` that are active for this user.
        category:
            Transaction category string (e.g. "dining", "travel").

        Returns
        -------
        dict with keys:
            ranked          — adjusted + re-sorted card list
            persona_context — human-readable summary of adjustments
        """
        valid = [p for p in active_personas if p in self._personas]

        if not valid:
            return {
                "ranked": ranked,
                "persona_context": "",
            }

        category_boost = self._merged_category_boost(valid, category)
        fee_multiplier = self._merged_fee_multiplier(valid)

        adjusted: List[Dict[str, Any]] = []
        for card in ranked:
            reward = float(card.get("reward_amount", 0.0))
            annual_fee = float(card.get("annual_fee", 0.0))

            boosted_reward = reward * category_boost
            extra_penalty = (fee_multiplier - 1.0) * annual_fee / 12.0
            adjusted_reward = boosted_reward - extra_penalty

            adjusted.append(
                {
                    **card,
                    "reward_amount": round(adjusted_reward, 4),
                    "persona_adjustments": {
                        "category_boost_applied": round(category_boost, 3),
                        "fee_multiplier_applied": round(fee_multiplier, 3),
                        "extra_fee_penalty": round(extra_penalty, 4),
                    },
                }
            )

        adjusted.sort(key=lambda c: c["reward_amount"], reverse=True)
        for i, card in enumerate(adjusted):
            card["rank"] = i + 1

        return {
            "ranked": adjusted,
            "persona_context": self._build_context(
                valid, category_boost, fee_multiplier
            ),
        }

    def persona_descriptions(self, active_personas: List[str]) -> List[str]:
        """Return description strings for the given active persona keys."""
        return [
            self._personas[p]["description"]
            for p in active_personas
            if p in self._personas
        ]

    def card_persona_reason(
        self,
        card: Dict[str, Any],
        active_personas: List[str],
        category: str,
    ) -> str:
        """Return a human-readable explanation of why *card* suits the personas.

        Parameters
        ----------
        card:
            A scored card dict (must include ``card_id``, ``card_name``,
            ``annual_fee``, and optionally ``persona_adjustments``).
        active_personas:
            Persona keys active for the current user.
        category:
            Transaction category being evaluated.

        Returns
        -------
        str — one or two sentence explanation.
        """
        valid = [p for p in active_personas if p in self._personas]
        if not valid:
            return "No active persona — ranked by raw reward value."

        parts: List[str] = []
        annual_fee = float(card.get("annual_fee", 0.0))
        adj = card.get("persona_adjustments") or {}

        category_boost = float(adj.get("category_boost_applied", 1.0))
        fee_multiplier = float(adj.get("fee_multiplier_applied", 1.0))

        descriptions = [self._personas[p].get("description", p) for p in valid]
        parts.append(f"Matches {', '.join(valid)} ({'; '.join(descriptions)}).")

        if category_boost > 1.0:
            parts.append(
                f"{category} rewards boosted ×{category_boost:.2f} for this persona."
            )

        if annual_fee == 0:
            if fee_multiplier > 1.0:
                parts.append("No annual fee — ideal for fee-sensitive personas.")
        elif fee_multiplier > 1.0:
            parts.append(
                f"${annual_fee:.0f} annual fee is penalised under fee-sensitive persona."
            )
        elif fee_multiplier < 1.0:
            parts.append(
                f"${annual_fee:.0f} annual fee is discounted — persona favours premium benefits."
            )

        return " ".join(parts)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _merged_category_boost(self, personas: List[str], category: str) -> float:
        """Average category boost across all active personas."""
        boosts = [
            float(self._personas[p].get("category_boosts", {}).get(category, 1.0))
            for p in personas
        ]
        return sum(boosts) / len(boosts)

    def _merged_fee_multiplier(self, personas: List[str]) -> float:
        """Average annual_fee_multiplier across all active personas."""
        multipliers = [
            float(self._personas[p].get("annual_fee_multiplier", 1.0)) for p in personas
        ]
        return sum(multipliers) / len(multipliers)

    def _build_context(
        self, personas: List[str], category_boost: float, fee_multiplier: float
    ) -> str:
        descriptions = [self._personas[p].get("description", p) for p in personas]
        parts = [f"Active personas: {', '.join(personas)}."]
        parts.append(" | ".join(descriptions))
        if category_boost != 1.0:
            parts.append(f"Category reward boosted ×{category_boost:.2f}.")
        if fee_multiplier != 1.0:
            direction = "increased" if fee_multiplier > 1.0 else "reduced"
            parts.append(f"Annual fee sensitivity {direction} (×{fee_multiplier:.2f}).")
        return " ".join(parts)
