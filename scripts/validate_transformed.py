# scripts/validate_transformed.py
from __future__ import annotations

from pathlib import Path

import pandas as pd


def latest_run_dir(root: Path) -> Path:
    runs = sorted([p for p in root.iterdir() if p.is_dir()])
    if not runs:
        raise FileNotFoundError(f"No run folders found under: {root}")
    return runs[-1]


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def print_nulls(df: pd.DataFrame, top: int = 10) -> None:
    na = (df.isna().mean() * 100).sort_values(ascending=False).head(top)
    print(na.round(2).to_string())


def check_duplicates(df: pd.DataFrame, col: str, label: str) -> None:
    if col not in df.columns:
        print(f"{label}: (skip) missing column {col}")
        return
    d = int(df[col].duplicated().sum())
    print(f"{label}: {d}")


def check_nonnegative(df: pd.DataFrame, col: str, label: str) -> None:
    if col not in df.columns:
        print(f"{label}: (skip) missing column {col}")
        return
    s = pd.to_numeric(df[col], errors="coerce")
    bad = int((s < 0).sum())
    print(f"{label}: {bad}")


def check_range(df: pd.DataFrame, col: str, lo: float, hi: float, label: str) -> None:
    if col not in df.columns:
        print(f"{label}: (skip) missing column {col}")
        return
    s = pd.to_numeric(df[col], errors="coerce")
    bad = int(((s < lo) | (s > hi)).sum())
    print(f"{label}: {bad} outside [{lo}, {hi}]")


def check_future_dates(df: pd.DataFrame, col: str, label: str) -> None:
    if col not in df.columns:
        print(f"{label}: (skip) missing column {col}")
        return
    dt = pd.to_datetime(df[col], errors="coerce")
    bad = int((dt > pd.Timestamp.now()).sum())
    print(f"{label}: {bad}")


def check_user_overlap(tx_features: pd.DataFrame, users: pd.DataFrame) -> None:
    if "user_id" not in tx_features.columns or "user_id" not in users.columns:
        print("user overlap: (skip) missing user_id")
        return
    tx_users = set(tx_features["user_id"].dropna().astype(str))
    us_users = set(users["user_id"].dropna().astype(str))
    print("users in tx_features not in users_features:", len(tx_users - us_users))
    print("users in users_features not in tx_features:", len(us_users - tx_users))


def main() -> int:
    root = Path("data/processed/current/transformed").resolve()
    run = latest_run_dir(root)
    final = run / "final"

    cards_path = final / "credit_cards_features.csv"
    tx_path = final / "transactions_features.csv"
    users_path = final / "users_features.csv"

    print("RUN:", run.name)
    print("FINAL DIR:", final)

    cards = load_csv(cards_path)
    txf = load_csv(tx_path)
    users = load_csv(users_path)

    print("\n== credit_cards_features.csv ==")
    print("shape:", cards.shape)
    print("null% top 10:")
    print_nulls(cards)

    print("\n== transactions_features.csv ==")
    print("shape:", txf.shape)
    print("null% top 10:")
    print_nulls(txf)

    print("\n== users_features.csv ==")
    print("shape:", users.shape)
    print("null% top 10:")
    print_nulls(users)

    print("\n--- Basic validity checks ---")
    check_duplicates(cards, "card_id", "cards.card_id duplicates")
    check_duplicates(users, "user_id", "users.user_id duplicates")
    check_duplicates(
        txf, "user_id", "tx_features.user_id duplicates (ok if multiple rows per user?)"
    )

    # Cards sanity
    check_nonnegative(cards, "annual_fee", "cards annual_fee < 0")
    check_nonnegative(cards, "base_reward_rate", "cards base_reward_rate < 0")
    check_range(cards, "base_reward_rate", 0, 50, "cards base_reward_rate")  # percent

    # Users sanity
    check_nonnegative(users, "monthly_budget", "users monthly_budget < 0")
    check_nonnegative(users, "num_cards", "users num_cards < 0")

    # If you still have raw txns somewhere and want to check date, point this to that file instead.
    # check_future_dates(tx_raw, "date", "txns future dates")

    print("\n--- Cross-dataset consistency ---")
    check_user_overlap(txf, users)

    print("\n✅ Validation complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
