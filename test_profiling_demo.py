"""
Demo script to test profiling module with real data.
"""

import pandas as pd
import json
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

from src.data_pipeline.profiling import (  # noqa: E402
    DataProfiler,
    CreditCardStatistics,
    TransactionStatistics,
    UserProfileStatistics,
    create_reward_distribution_plot,
    create_spending_heatmap,
    StatisticsHistory,
)

print("=" * 70)
print("RewardSense Profiling Module Demo")
print("=" * 70)

# =====================================================================
# Load Data
# =====================================================================

print("\n[1/5] Loading data...")

# Transactions
txns_df = pd.read_csv("data/processed/current/synthetic/transactions.csv")
print(f"✅ Loaded {len(txns_df)} transactions")

# Users
users_df = pd.read_csv("data/processed/current/synthetic/user_profiles.csv")
print(f"✅ Loaded {len(users_df)} users")

# Credit cards
with open("data/processed/current/offers/creditcardbonuses_offers.json", "r") as f:
    cc_data = json.load(f)
cards_df = pd.DataFrame(cc_data["offers"])
print(f"✅ Loaded {len(cards_df)} credit cards")

# =====================================================================
# Generate Custom Statistics
# =====================================================================

print("\n[2/5] Generating custom statistics...")

# Transaction stats
txn_stats = TransactionStatistics.calculate_statistics(txns_df)
print("✅ Transaction stats:")
print(f"   Total spending: ${txn_stats['amount']['total']:,.2f}")
print(f"   Avg transaction: ${txn_stats['amount']['mean']:.2f}")
print(f"   Categories: {len(txn_stats['category_distribution'])}")

# User stats
user_stats = UserProfileStatistics.calculate_statistics(users_df)
print("✅ User profile stats:")
print(f"   Total users: {user_stats['total_users']}")
print(f"   Avg monthly budget: ${user_stats['monthly_budget']['mean']:,.2f}")
print(f"   Archetypes: {len(user_stats['archetype_distribution'])}")

# Credit card stats
cc_stats = CreditCardStatistics.calculate_statistics(cards_df)
print("✅ Credit card stats:")
print(f"   Total cards: {cc_stats['total_cards']}")
print(f"   No-fee cards: {cc_stats['annual_fee']['no_fee_pct']:.1f}%")
print(f"   Premium cards: {cc_stats['annual_fee']['premium_pct']:.1f}%")

# =====================================================================
# Save Historical Statistics
# =====================================================================

print("\n[3/5] Saving historical statistics...")

history = StatisticsHistory(history_dir="data/profiling/history")

history.save_statistics("transactions", txn_stats)
history.save_statistics("users", user_stats)
history.save_statistics("cards", cc_stats)

print("✅ Statistics saved to history")

# =====================================================================
# Generate Visualizations
# =====================================================================

print("\n[4/5] Generating visualizations...")

viz_dir = Path("data/profiling/visualizations")
viz_dir.mkdir(parents=True, exist_ok=True)

# Reward distribution
create_reward_distribution_plot(
    cards_df, output_path=viz_dir / "reward_distribution.png"
)
print("✅ Created reward distribution plot")

# Spending patterns
create_spending_heatmap(txns_df, output_path=viz_dir / "spending_by_category.png")
print("✅ Created spending heatmap")

# =====================================================================
# Generate Minimal Profile Reports (Fast)
# =====================================================================

print("\n[5/5] Generating ydata-profiling reports (minimal mode)...")

profiler = DataProfiler(output_dir="data/profiling/reports")

# Generate minimal profiles (fast)
txn_profile = profiler.profile_transactions(txns_df.head(1000), minimal=True)
print("✅ Generated transaction profile (1000 samples)")

user_profile = profiler.profile_user_profiles(users_df, minimal=True)
print("✅ Generated user profile")

print("\n" + "=" * 70)
print("PROFILING COMPLETE!")
print("=" * 70)
print("\n📊 Outputs:")
print("   Statistics: data/profiling/history/")
print("   Visualizations: data/profiling/visualizations/")
print("   Profile reports: data/profiling/reports/")
print("\n" + "=" * 70)
