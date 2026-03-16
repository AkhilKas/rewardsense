# RewardSense Field Catalog (Compact)

This compact catalog explains what each final dataset contains without listing every column one-by-one.

- Source directory: `data/processed/current/transformed/20260224_020359/final`
- Generated at (UTC): `2026-03-14T17:08:01Z`

## `credit_cards_features.csv`

- Total columns: **159**
- Feature families:
  - **Business flags and lifecycle**: 8 columns
    Examples: `is_business`, `is_annual_fee_waived`, `discontinued`, `is_active`, `is_discontinued`, `is_premium`
  - **Flattened reward-rate details**: 27 columns
    Examples: `reward_rates.chase travel purchases.rate`, `reward_rates.chase travel purchases.type`, `reward_rates.other travel purchases.rate`, `reward_rates.other travel purchases.type`, `reward_rates.is just the beginning.rate`, `reward_rates.is just the beginning.type`
  - **Identifiers**: 1 columns
    Examples: `card_id`
  - **Card economics/value engineered features**: 11 columns
    Examples: `bonus_difficulty`, `annual_credits_value`, `num_credits`, `has_credits`, `effective_annual_fee`, `effective_fee_year1`
  - **Raw source traceability**: 16 columns
    Examples: `raw.cardId`, `raw.name`, `raw.issuer`, `raw.network`, `raw.currency`, `raw.isBusiness`
  - **Currency/reward-currency one-hot encodings**: 41 columns
    Examples: `currency_ALASKA`, `currency_AMERICAN`, `currency_AMERICAN_EXPRESS`, `currency_AMTRAK`, `currency_ANA`, `currency_AVIANCA`
  - **Issuer one-hot encodings**: 21 columns
    Examples: `issuer_source`, `issuer_original`, `issuer_clean`, `issuer_AMERICAN EXPRESS`, `issuer_BANK OF AMERICA`, `issuer_BARCLAYS`
  - **Core attributes**: 22 columns
    Examples: `source`, `scraped_at`, `name`, `issuer`, `detail_url`, `annual_fee`
  - **Welcome bonus engineered features**: 8 columns
    Examples: `welcome_bonus_parsed`, `welcome_bonus_amount`, `welcome_bonus_unit`, `welcome_bonus_spend_req`, `welcome_bonus_days`, `welcome_bonus_value_cents`
  - **Network one-hot encodings**: 4 columns
    Examples: `network_AMERICAN_EXPRESS`, `network_DISCOVER`, `network_MASTERCARD`, `network_VISA`


## `transactions_features.csv`

- Total columns: **48**
- Feature families:
  - **Identifiers**: 1 columns
    Examples: `user_id`
  - **Card usage, merchant, MCC, and anomaly behavior**: 11 columns
    Examples: `num_cards_used`, `primary_card`, `card_switch_rate`, `num_unique_mccs`, `primary_mcc`, `avg_spending_per_mcc`
  - **Spend and transaction behavior aggregates**: 36 columns
    Examples: `dining_total_spent`, `drugstore_total_spent`, `entertainment_total_spent`, `gas_total_spent`, `groceries_total_spent`, `home_improvement_total_spent`

## `users_features.csv`

- Total columns: **39**
- Feature families:
  - **Identifiers**: 1 columns
    Examples: `user_id`
  - **User segmentation and preference engineered features**: 33 columns
    Examples: `redemption_preference`, `age_group`, `location_type`, `archetype_budget_conscious`, `archetype_category_specialist`, `archetype_frequent_traveler`
  - **Core attributes**: 5 columns
    Examples: `archetype`, `monthly_budget`, `cards`, `cards_list`, `num_cards`

## Practical Interpretation

- The wide schema is intentional: it combines explainability fields + model-ready numeric features.
- For training, you typically exclude IDs/high-cardinality text and keep engineered + encoded columns.
- For analytics/demo, retain human-readable core attributes and top engineered value features.
