import pandas as pd
from pathlib import Path

def aggregate_district_daily(df:pd.DataFrame):
    agg = (
        df.groupby(["date", "state", "district"], as_index=False).agg(youth=("demo_age_5_17", "sum"), adult = ("demo_age_17_", "sum"))
    )

    agg["total"] = agg["youth"] + agg["adult"]

    agg = agg.sort_values(["state", "district", "date"]).reset_index(drop=True)

    return agg

def remove_invalid_geography(df: pd.DataFrame) -> pd.DataFrame:

    mask_valid = (
        df["state"].str.contains(r"[a-zA-Z]", regex=True) &
        df["district"].str.contains(r"[a-zA-Z]", regex=True)
    )

    removed = len(df) - mask_valid.sum()
    print(f"Removed {removed} rows with invalid geography")

    return df[mask_valid]


def main():
    df = pd.read_parquet("/Users/muaazshaikh/aadhar-forecast-anomaly/data/raw_combined.parquet")
    agg_df = aggregate_district_daily(df)
    agg_df = remove_invalid_geography(agg_df)


    assert agg_df.isna().sum().sum() == 0, "Unexpected NaNs after aggregation"
    assert (agg_df[["youth", "adult", "total"]] < 0).sum().sum() == 0, "Negative counts found"

    agg_df.to_parquet("/Users/muaazshaikh/aadhar-forecast-anomaly/data/agg_district_daily.parquet", index = False)
    print("Shape:", agg_df.shape)
    print("Date range:", agg_df["date"].min(), "->", agg_df["date"].max())
    print("Unique districts:", agg_df[["state", "district"]].drop_duplicates().shape[0])


if __name__ == "__main__":
    main()