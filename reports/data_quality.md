# Data Quality Report - Aadhaar Demographic Dataset

## Dataset Overview
- Rows: ~1.4M
- Columns: 6
- Time range: 2025-01-03 to 2025-12-29

## Missing Values
- Core keys (date/state/district): negligible
- Counts: low, handled downstream

## Duplicates
- Exact duplicates removed during ingestion
- Multiple geo-date rows expected (handled via aggregation)

## Temporal Coverage
- Non-continuous daily records
- Suitable for aggregation based time series ML

## Geographic Integrity
- ~58 states
- District names reused across states -> composite key used

## Value Sanity
- No negative counts
- Long tailed distributions consistent with real world data
