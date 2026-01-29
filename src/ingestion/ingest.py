import pandas as pd
import numpy as np 
from pathlib import Path

FILES = [
    "/Users/muaazshaikh/aadhar-forecast-anomaly/data/api_data_aadhar_demographic_0_500000.csv",
    "/Users/muaazshaikh/aadhar-forecast-anomaly/data/api_data_aadhar_demographic_500000_1000000.csv",
    "/Users/muaazshaikh/aadhar-forecast-anomaly/data/api_data_aadhar_demographic_1000000_1500000.csv",
    "/Users/muaazshaikh/aadhar-forecast-anomaly/data/api_data_aadhar_demographic_1500000_2000000.csv",
    "/Users/muaazshaikh/aadhar-forecast-anomaly/data/api_data_aadhar_demographic_2000000_2071700.csv"
]

OUT_DIR = Path("data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DTYPES = {
    "state": "string",
    "district": "string",
    "pincode": "string",
    "demo_age_5_17": "Int32",
    "demo_age_17_": "Int32",
}

REQUIRED_COLS = {"date", "state", "district", "pincode", "demo_age_5_17", "demo_age_17_"}

# first lets just convert the csvs into a clean dataset

def clean_chunk(df: pd.DataFrame):
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"missing columns {missing}")
    
    df["date"] = pd.to_datetime(df["date"], errors = "coerce")

    for col in ["state", "district", "pincode"]:
        df[col] = df[col].astype("string").str.strip()

    df["state"] = df["state"].str.lower()
    df["district"] = df["district"].str.lower()

    df["pincode"] = df["pincode"].str.replace(r"\s+", "",regex=True)

    for c in ["demo_age_5_17", "demo_age_17_"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int32")
        # negative counts are set to NA (or 0; but NA is safer for QA)
        df.loc[df[c] < 0, c] = pd.NA

    return df


# ingesting the dataset into a single parquet (for context parquet is also a column oriented data file format for data storage whilst csv is row)

def ingest_to_parquet(files, out_path="data/raw_combined.parquet", chunksize=250_000):
    parts = []
    qa = {
        "rows_read": 0,
        "rows_kept": 0,
        "bad_date_rows": 0,
        "missing_state": 0,
        "missing_district": 0,
        "missing_pincode": 0,
        "missing_youth": 0,
        "missing_adult": 0,
    }

    for fp in files:
        for chunk in pd.read_csv(fp, dtype=DTYPES, chunksize=chunksize):
            qa["rows_read"] += len(chunk)

            chunk = clean_chunk(chunk)

            # basic missingness counters
            qa["bad_date_rows"] += chunk["date"].isna().sum()
            qa["missing_state"] += chunk["state"].isna().sum()
            qa["missing_district"] += chunk["district"].isna().sum()
            qa["missing_pincode"] += chunk["pincode"].isna().sum()
            qa["missing_youth"] += chunk["demo_age_5_17"].isna().sum()
            qa["missing_adult"] += chunk["demo_age_17_"].isna().sum()

            # drop rows where core keys are missing
            chunk = chunk.dropna(subset=["date", "state", "district"])

            qa["rows_kept"] += len(chunk)
            parts.append(chunk)

    df = pd.concat(parts, ignore_index=True)

    # drop exact duplicates (after standardization)
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)

    # convert to categories 
    df["state"] = df["state"].astype("category")
    df["district"] = df["district"].astype("category")

    # save
    df.to_parquet(out_path, index=False)

    # quick QA summary
    summary = {
        **qa,
        "duplicates_removed": before - after,
        "date_min": str(df["date"].min()),
        "date_max": str(df["date"].max()),
        "final_rows": len(df),
        "final_cols": list(df.columns),
    }
    pd.Series(summary).to_json("data/ingestion_qa.json", indent=2)
    return df, summary

if __name__ == "__main__":
    df, summary = ingest_to_parquet(FILES)
    print("Saved:", "data/raw_combined.parquet")
    print("QA Summary:", summary)