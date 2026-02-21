"""
Main data profiling module using ydata-profiling.

Generates comprehensive data quality reports for all datasets.
"""

import pandas as pd
from ydata_profiling import ProfileReport
from pathlib import Path
import logging
from typing import Optional, Dict
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class DataProfiler:
    """
    Generate data profiles using ydata-profiling.

    Creates comprehensive HTML and JSON reports with:
    - Summary statistics
    - Missing value analysis
    - Distribution plots
    - Correlation matrices
    - Duplicate detection
    """

    def __init__(self, output_dir: str = "data/profiling"):
        """
        Initialize the data profiler.

        Args:
            output_dir: Directory to save profile reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized DataProfiler with output_dir: {self.output_dir}")

    def profile_transactions(
        self,
        df: pd.DataFrame,
        title: str = "Transaction Data Profile",
        minimal: bool = False,
    ) -> ProfileReport:
        """
        Generate profile report for transaction data.

        Args:
            df: Transaction DataFrame
            title: Report title
            minimal: If True, generate minimal report (faster)

        Returns:
            ProfileReport object
        """
        logger.info(f"Generating transaction profile for {len(df)} records...")

        config = {
            "title": title,
            "minimal": minimal,
            "correlations": {
                "auto": {"calculate": True},
            },
            "missing_diagrams": {
                "bar": True,
                "matrix": True,
            },
        }

        profile = ProfileReport(df, **config)

        # Save HTML report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_path = self.output_dir / f"transactions_profile_{timestamp}.html"
        profile.to_file(html_path)
        logger.info(f"Saved transaction profile to: {html_path}")

        # Save JSON summary
        json_path = self.output_dir / f"transactions_profile_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump(profile.get_description(), f, indent=2, default=str)

        return profile

    def profile_user_profiles(
        self,
        df: pd.DataFrame,
        title: str = "User Profile Data Profile",
        minimal: bool = False,
    ) -> ProfileReport:
        """
        Generate profile report for user profile data.

        Args:
            df: User profile DataFrame
            title: Report title
            minimal: If True, generate minimal report

        Returns:
            ProfileReport object
        """
        logger.info(f"Generating user profile for {len(df)} records...")

        config = {
            "title": title,
            "minimal": minimal,
            "correlations": {
                "auto": {"calculate": True},
            },
        }

        profile = ProfileReport(df, **config)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_path = self.output_dir / f"user_profiles_profile_{timestamp}.html"
        profile.to_file(html_path)
        logger.info(f"Saved user profile to: {html_path}")

        json_path = self.output_dir / f"user_profiles_profile_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump(profile.get_description(), f, indent=2, default=str)

        return profile

    def profile_credit_cards(
        self,
        df: pd.DataFrame,
        title: str = "Credit Card Data Profile",
        minimal: bool = False,
    ) -> ProfileReport:
        """
        Generate profile report for credit card data.

        Args:
            df: Credit card DataFrame
            title: Report title
            minimal: If True, generate minimal report

        Returns:
            ProfileReport object
        """
        logger.info(f"Generating credit card profile for {len(df)} records...")

        config = {
            "title": title,
            "minimal": minimal,
        }

        profile = ProfileReport(df, **config)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_path = self.output_dir / f"credit_cards_profile_{timestamp}.html"
        profile.to_file(html_path)
        logger.info(f"Saved credit card profile to: {html_path}")

        json_path = self.output_dir / f"credit_cards_profile_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump(profile.get_description(), f, indent=2, default=str)

        return profile

    def profile_all(
        self,
        transactions_df: Optional[pd.DataFrame] = None,
        users_df: Optional[pd.DataFrame] = None,
        cards_df: Optional[pd.DataFrame] = None,
        minimal: bool = False,
    ) -> Dict[str, ProfileReport]:
        """
        Generate profiles for all datasets.

        Args:
            transactions_df: Transaction DataFrame
            users_df: User profile DataFrame
            cards_df: Credit card DataFrame
            minimal: Generate minimal reports (faster)

        Returns:
            Dictionary of {dataset_name: ProfileReport}
        """
        logger.info("Generating profiles for all datasets...")

        profiles = {}

        if transactions_df is not None:
            profiles["transactions"] = self.profile_transactions(
                transactions_df, minimal=minimal
            )

        if users_df is not None:
            profiles["users"] = self.profile_user_profiles(users_df, minimal=minimal)

        if cards_df is not None:
            profiles["cards"] = self.profile_credit_cards(cards_df, minimal=minimal)

        logger.info(f"Generated {len(profiles)} profile reports")
        return profiles


# Convenience function
def generate_profile_report(
    df: pd.DataFrame,
    title: str = "Data Profile",
    output_path: Optional[Path] = None,
    minimal: bool = False,
) -> ProfileReport:
    """
    Generate a profile report for any DataFrame.

    Args:
        df: DataFrame to profile
        title: Report title
        output_path: Optional path to save HTML report
        minimal: Generate minimal report (faster)

    Returns:
        ProfileReport object
    """
    profile = ProfileReport(df, title=title, minimal=minimal)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        profile.to_file(output_path)
        logger.info(f"Saved profile to: {output_path}")

    return profile
