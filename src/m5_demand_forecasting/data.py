"""Build the California subset of the raw M5 files."""

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

STATE = "CA"
ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]


def build_state_subset(
    raw_dir: Path = RAW_DIR, out_dir: Path = PROCESSED_DIR, state: str = STATE
) -> None:
    """Filter sales and prices to one state and save them, with the calendar, as parquet."""
    out_dir.mkdir(parents=True, exist_ok=True)

    sales = pd.read_csv(raw_dir / "sales_train_evaluation.csv")
    sales = sales[sales["state_id"] == state].reset_index(drop=True)
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    sales[day_cols] = sales[day_cols].astype("int16")
    sales.to_parquet(out_dir / "sales_wide.parquet", index=False)

    prices = pd.read_csv(raw_dir / "sell_prices.csv")
    prices = prices[prices["store_id"].str.startswith(f"{state}_")].reset_index(drop=True)
    prices.to_parquet(out_dir / "prices.parquet", index=False)

    calendar = pd.read_csv(raw_dir / "calendar.csv", parse_dates=["date"])
    calendar.to_parquet(out_dir / "calendar.parquet", index=False)

    print(f"sales:    {sales.shape[0]:,} series x {len(day_cols):,} days")
    print(f"prices:   {len(prices):,} rows")
    print(f"calendar: {len(calendar):,} days")


if __name__ == "__main__":
    build_state_subset()
