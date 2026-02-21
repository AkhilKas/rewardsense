"""
Data profiling module for RewardSense.

Provides automated profiling, statistics generation, and visualization
for credit card, transaction, and user profile data.
"""

from .profiler import DataProfiler, generate_profile_report
from .statistics import (
    CreditCardStatistics,
    TransactionStatistics,
    UserProfileStatistics,
)
from .visualizations import (
    create_reward_distribution_plot,
    create_spending_heatmap,
    create_temporal_trend_plot,
    generate_all_visualizations,
)
from .history import StatisticsHistory

__all__ = [
    "DataProfiler",
    "generate_profile_report",
    "CreditCardStatistics",
    "TransactionStatistics",
    "UserProfileStatistics",
    "create_reward_distribution_plot",
    "create_spending_heatmap",
    "create_temporal_trend_plot",
    "generate_all_visualizations",
    "StatisticsHistory",
]
