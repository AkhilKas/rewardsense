"""
Custom visualizations for RewardSense data quality and domain metrics.

Creates specialized plots for:
- Credit card reward distributions
- Spending patterns by category
- User behavior heatmaps
- Temporal trends
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)


def create_reward_distribution_plot(
    df: pd.DataFrame, output_path: Optional[Path] = None
) -> plt.Figure:
    """
    Create distribution plot of credit card reward rates.

    Args:
        df: Credit card DataFrame with base_reward_rate column
        output_path: Optional path to save plot

    Returns:
        Matplotlib figure
    """
    logger.info("Creating reward distribution plot...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Credit Card Reward Structure Analysis", fontsize=16, fontweight="bold"
    )

    # Plot 1: Base reward rate distribution
    if "base_reward_rate" in df.columns:
        axes[0, 0].hist(df["base_reward_rate"], bins=20, edgecolor="black", alpha=0.7)
        axes[0, 0].set_xlabel("Base Reward Rate (%)")
        axes[0, 0].set_ylabel("Number of Cards")
        axes[0, 0].set_title("Base Reward Rate Distribution")
        axes[0, 0].axvline(
            df["base_reward_rate"].mean(),
            color="red",
            linestyle="--",
            label=f"Mean: {df['base_reward_rate'].mean():.2f}%",
        )
        axes[0, 0].legend()

    # Plot 2: Annual fee distribution
    if "annual_fee" in df.columns:
        axes[0, 1].hist(df["annual_fee"], bins=30, edgecolor="black", alpha=0.7)
        axes[0, 1].set_xlabel("Annual Fee ($)")
        axes[0, 1].set_ylabel("Number of Cards")
        axes[0, 1].set_title("Annual Fee Distribution")
        axes[0, 1].axvline(
            df["annual_fee"].median(),
            color="red",
            linestyle="--",
            label=f"Median: ${df['annual_fee'].median():.0f}",
        )
        axes[0, 1].legend()

    # Plot 3: Issuer distribution
    if "issuer" in df.columns:
        issuer_counts = df["issuer"].value_counts().head(10)
        axes[1, 0].barh(range(len(issuer_counts)), issuer_counts.values)
        axes[1, 0].set_yticks(range(len(issuer_counts)))
        axes[1, 0].set_yticklabels(issuer_counts.index)
        axes[1, 0].set_xlabel("Number of Cards")
        axes[1, 0].set_title("Top 10 Issuers")

    # Plot 4: Net value distribution (if features calculated)
    if "net_value_annual" in df.columns:
        axes[1, 1].hist(df["net_value_annual"], bins=30, edgecolor="black", alpha=0.7)
        axes[1, 1].set_xlabel("Net Annual Value ($)")
        axes[1, 1].set_ylabel("Number of Cards")
        axes[1, 1].set_title("Net Annual Value Distribution")
        axes[1, 1].axvline(0, color="red", linestyle="--", label="Break-even")
        axes[1, 1].legend()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved reward distribution plot to: {output_path}")

    return fig


def create_spending_heatmap(
    df: pd.DataFrame, output_path: Optional[Path] = None
) -> plt.Figure:
    """
    Create heatmap of spending patterns by category and user archetype.

    Args:
        df: Transaction DataFrame with category and amount columns
        output_path: Optional path to save plot

    Returns:
        Matplotlib figure
    """
    logger.info("Creating spending heatmap...")

    # This requires joining with user data - simplified version
    if "category" in df.columns and "amount" in df.columns:
        # Aggregate by category
        category_spending = df.groupby("category")["amount"].agg(
            ["sum", "mean", "count"]
        )

        fig, ax = plt.subplots(figsize=(12, 6))

        # Create bar plot
        x = range(len(category_spending))
        ax.bar(x, category_spending["sum"], alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(category_spending.index, rotation=45, ha="right")
        ax.set_xlabel("Category")
        ax.set_ylabel("Total Spending ($)")
        ax.set_title("Total Spending by Category")

        # Add mean line
        ax2 = ax.twinx()
        ax2.plot(
            x,
            category_spending["mean"],
            color="red",
            marker="o",
            label="Avg Transaction",
        )
        ax2.set_ylabel("Average Transaction Amount ($)", color="red")
        ax2.tick_params(axis="y", labelcolor="red")
        ax2.legend(loc="upper right")

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved spending heatmap to: {output_path}")

        return fig

    logger.warning("Required columns not found for spending heatmap")
    return None


def create_temporal_trend_plot(
    df: pd.DataFrame, output_path: Optional[Path] = None
) -> go.Figure:
    """
    Create interactive temporal spending trend plot using Plotly.

    Args:
        df: Transaction DataFrame with date and amount columns
        output_path: Optional path to save HTML plot

    Returns:
        Plotly figure
    """
    logger.info("Creating temporal trend plot...")

    if "date" in df.columns and "amount" in df.columns:
        df_temp = df.copy()
        df_temp["date"] = pd.to_datetime(df_temp["date"])

        # Aggregate by date
        daily_spending = df_temp.groupby("date")["amount"].sum().reset_index()
        daily_spending.columns = ["date", "total_amount"]

        # Create Plotly figure
        fig = px.line(
            daily_spending,
            x="date",
            y="total_amount",
            title="Daily Spending Over Time",
            labels={"total_amount": "Total Spending ($)", "date": "Date"},
        )

        fig.update_layout(hovermode="x unified", template="plotly_white")

        if output_path:
            fig.write_html(output_path)
            logger.info(f"Saved temporal trend plot to: {output_path}")

        return fig

    logger.warning("Required columns not found for temporal plot")
    return None


def generate_all_visualizations(
    transactions_df: Optional[pd.DataFrame] = None,
    users_df: Optional[pd.DataFrame] = None,
    cards_df: Optional[pd.DataFrame] = None,
    output_dir: str = "data/profiling/visualizations",
) -> Dict[str, Any]:
    """
    Generate all custom visualizations.

    Args:
        transactions_df: Transaction DataFrame
        users_df: User profile DataFrame
        cards_df: Credit card DataFrame
        output_dir: Output directory for plots

    Returns:
        Dictionary of generated plot paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    plots = {}

    if cards_df is not None:
        plot_path = output_path / "reward_distribution.png"
        create_reward_distribution_plot(cards_df, plot_path)
        plots["reward_distribution"] = str(plot_path)

    if transactions_df is not None:
        plot_path = output_path / "spending_heatmap.png"
        create_spending_heatmap(transactions_df, plot_path)
        plots["spending_heatmap"] = str(plot_path)

        plot_path = output_path / "temporal_trends.html"
        create_temporal_trend_plot(transactions_df, plot_path)
        plots["temporal_trends"] = str(plot_path)

    logger.info(f"Generated {len(plots)} visualizations in {output_dir}")
    return plots
