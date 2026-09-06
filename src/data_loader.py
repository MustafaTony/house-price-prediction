"""
data_loader.py

Loads the REAL house_prices.csv (Juhi Bhojani, Kaggle) and provides
schema-inspection utilities. Does NOT generate or fabricate any data.

If data/house_prices.csv is missing, functions here raise a clear error
rather than silently falling back to fake/synthetic data.
"""

from pathlib import Path
import pandas as pd

DEFAULT_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "house_prices.csv"


def load_real_dataset(path: Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """
    Load the real Kaggle house_prices.csv.

    Raises FileNotFoundError with an explicit message if the real file
    isn't present -- this project must never substitute synthetic data.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Real dataset not found at '{path}'.\n"
            "This project requires the actual house_prices.csv from Kaggle "
            "(House Price dataset by Juhi Bhojani). Place the file there and "
            "re-run. No synthetic data will be substituted."
        )
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"'{path}' was found but contains no rows.")
    return df


def inspect_schema(df: pd.DataFrame) -> dict:
    """
    Returns a summary of the real dataframe's schema:
    - column names & dtypes
    - missing value counts
    - number of unique values per column
    - basic numeric stats (via describe)

    Use this output to fill in CONFIG (target column, location column)
    in the notebook / train_model.py once you have the real file.
    """
    summary = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "n_unique": df.nunique().to_dict(),
    }
    return summary


def guess_target_column(df: pd.DataFrame) -> str | None:
    """
    Best-effort guess at the price/target column based on common naming
    patterns seen in Kaggle house-price datasets. This is only a
    suggestion -- always confirm against the real column list printed
    by inspect_schema() before trusting it.
    """
    candidates = [
        "price", "Price", "PRICE",
        "saleprice", "SalePrice", "sale_price",
        "house_price", "HousePrice", "target",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: look for any column containing "price"
    for c in df.columns:
        if "price" in c.lower():
            return c
    return None


def guess_location_column(df: pd.DataFrame) -> str | None:
    """
    Best-effort guess at a location/area/city categorical column.
    Only a suggestion -- confirm against the real schema.
    """
    candidates = ["location", "Location", "city", "City", "area", "Area",
                  "neighborhood", "Neighborhood", "suburb", "Suburb"]
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        if any(key in c.lower() for key in ["location", "city", "area", "suburb", "neighborhood"]):
            return c
    return None


if __name__ == "__main__":
    df = load_real_dataset()
    summary = inspect_schema(df)
    print("Shape:", summary["shape"])
    print("\nColumns & dtypes:")
    for col, dtype in summary["dtypes"].items():
        print(f"  {col}: {dtype}  (missing={summary['missing_values'][col]}, "
              f"unique={summary['n_unique'][col]})")
    print("\nGuessed target column:", guess_target_column(df))
    print("Guessed location column:", guess_location_column(df))
