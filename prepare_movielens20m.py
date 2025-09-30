# prepare_movielens20m.py
import os, csv
import pandas as pd

INP = "data/Movielens-20M/rating.csv"   # adjust if your path differs
OUT = "data/Movielens-20M/ratings.csv"

def main():
    if not os.path.exists(INP):
        raise FileNotFoundError(f"Cannot find {INP}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    # Common ML-20M header: userId,movieId,rating,timestamp
    df = pd.read_csv(INP)
    # keep positives only (implicit feedback style)
    if "rating" in df.columns:
        df = df[df["rating"] >= 4].copy()
    # rename movieId -> itemId if needed
    if "movieId" in df.columns and "itemId" not in df.columns:
        df = df.rename(columns={"movieId": "itemId"})
    # ensure columns present
    df = df[["userId", "itemId", "rating", "timestamp"]]
    df.to_csv(OUT, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"Wrote {len(df)} rows to {OUT}")

if __name__ == "__main__":
    main()
