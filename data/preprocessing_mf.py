"""
Preprocess MovieLens ratings for Matrix Factorization.
Main purpose:
- get sequential idx
- proper test/train split (on timestamp, per user)

Steps
---------------------
1) Reads ratings.csv in chunk.
2) Optionally filter to reduce size
3) Re-index:
   - userId -> user_idx (contiguous, int32)  [NOT saved]
   - movieId -> item_idx (contiguous, int32) [SAVED, because we  need to map embeddings back to movies]
4) Build a per-user holdout split (last rating by timestamp by default):
   - train.csv
   - val.csv
   - items.csv (mapping item_idx -> movieId, title, genres)

Outputs (default)
-----------------
processed/
  train.csv: columns [user_idx, item_idx, rating]
  val.csv:   columns [user_idx, item_idx, rating]
  items.csv: columns [item_idx, movieId, title, genres]
"""
import argparse
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def read_ratings_in_chunks(ratings_path: str, chunksize: int) -> pd.DataFrame:
    """Read ratings.csv chunkwise and concatenate."""
    usecols = ["userId", "movieId", "rating", "timestamp"]
    chunks = []
    for chunk in pd.read_csv(ratings_path, usecols=usecols, chunksize=chunksize):
        # enforce dtypes early
        chunk["userId"] = chunk["userId"].astype(np.int64)
        chunk["movieId"] = chunk["movieId"].astype(np.int64)
        chunk["rating"] = chunk["rating"].astype(np.float32)
        chunk["timestamp"] = chunk["timestamp"].astype(np.int64)
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True)


def filter_ratings(
    df: pd.DataFrame,
    min_user_ratings: int,
    min_movie_ratings: int,
    top_k_movies: Optional[int],
    sample_frac: Optional[float],
    seed: int,
) -> pd.DataFrame:
    """
    Reduce dataset size while preserving useful structure.
    """
    out = df

    if top_k_movies is not None:
        movie_counts = out["movieId"].value_counts()
        keep_movies = movie_counts.head(int(top_k_movies)).index
        out = out[out["movieId"].isin(keep_movies)]

    if min_movie_ratings > 1:
        movie_counts = out["movieId"].value_counts()
        keep_movies = movie_counts[movie_counts >= int(min_movie_ratings)].index
        out = out[out["movieId"].isin(keep_movies)]

    if min_user_ratings > 1:
        user_counts = out["userId"].value_counts()
        keep_users = user_counts[user_counts >= int(min_user_ratings)].index
        out = out[out["userId"].isin(keep_users)]

    if sample_frac is not None and 0 < float(sample_frac) < 1:
        out = out.sample(frac=float(sample_frac), random_state=int(seed)).reset_index(drop=True)

    return out.reset_index(drop=True)


def per_user_last_timestamp_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Validation = last interaction per user (by timestamp).
    Train = all others.

    This guarantees:
    - every user appears in train (if the user has >=2 ratings)
    - no unknown userIds in val relative to train
    """
    # Sort by user then timestamp so "last" is deterministic
    df_sorted = df.sort_values(["userId", "timestamp"], ascending=[True, True]).reset_index(drop=True)

    # Mark last row per user as validation
    last_idx = df_sorted.groupby("userId", sort=False).tail(1).index
    val_df = df_sorted.loc[last_idx].copy()
    train_df = df_sorted.drop(index=last_idx).copy()

    # Users with only 1 rating would yield empty train rows for that user (problematic)
    # We already recommend filtering min_user_ratings >= 2
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def build_contiguous_indices(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create contiguous indices:
    - userId -> user_idx (NOT meant to be stable across preprocess runs)
    - movieId -> item_idx (we will save item mapping)
    """
    # movie mapping (stable within this processed dataset)
    unique_movies = np.sort(df["movieId"].unique())
    movie_to_idx = {int(mid): int(i) for i, mid in enumerate(unique_movies)}
    items_df = pd.DataFrame({"item_idx": np.arange(len(unique_movies), dtype=np.int32), "movieId": unique_movies})

    # user mapping
    unique_users = np.sort(df["userId"].unique())
    user_to_idx = {int(uid): int(i) for i, uid in enumerate(unique_users)}

    mapped = df.copy()
    mapped["item_idx"] = mapped["movieId"].map(movie_to_idx).astype(np.int32)
    mapped["user_idx"] = mapped["userId"].map(user_to_idx).astype(np.int32)

    # Keep only what MF training needs (+ timestamp only used for splitting earlier)
    mapped = mapped[["user_idx", "item_idx", "rating"]].copy()
    mapped["rating"] = mapped["rating"].astype(np.float32)

    return mapped, items_df


def maybe_join_movies_metadata(items_df: pd.DataFrame, movies_path: Optional[str]) -> pd.DataFrame:
    if not movies_path:
        return items_df
    movies_df = pd.read_csv(movies_path, usecols=["movieId", "title", "genres"])
    movies_df["movieId"] = movies_df["movieId"].astype(np.int64)
    out = items_df.merge(movies_df, on="movieId", how="left")
    return out


def save_csv(df: pd.DataFrame, path_base: str) -> str:
    """
    Save as parquet if possible, else csv.
    Returns actual path written.
    """
    csv_path = path_base if path_base.endswith(".csv") else path_base + ".csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings", required=True, help="Path to ratings.csv")
    ap.add_argument("--movies", default=None, help="Optional path to movies.csv (for title/genres join)")
    ap.add_argument("--outdir", default="processed", help="Output directory")

    ap.add_argument("--chunksize", type=int, default=2_000_000, help="Rows per chunk when reading CSV")

    # Size reduction knobs
    ap.add_argument("--min-user-ratings", type=int, default=20, help="Keep users with at least this many ratings")
    ap.add_argument("--min-movie-ratings", type=int, default=20, help="Keep movies with at least this many ratings")
    ap.add_argument("--top-k-movies", type=int, default=None, help="Keep only top-K movies by #ratings")
    ap.add_argument("--sample-frac", type=float, default=None, help="Optional random sample fraction (0..1)")
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print(f"Reading ratings (chunked): {args.ratings}")
    ratings_df = read_ratings_in_chunks(args.ratings, chunksize=args.chunksize)
    print(f"Loaded rows: {len(ratings_df):,}")

    # Enforce min_user_ratings >= 2 for last-timestamp split to make sense
    if args.min_user_ratings < 2:
        print("NOTE: setting --min-user-ratings to 2 (needed for per-user last-rating split).")
        args.min_user_ratings = 2

    print("Filtering ratings...")
    filtered = filter_ratings(
        ratings_df,
        min_user_ratings=args.min_user_ratings,
        min_movie_ratings=args.min_movie_ratings,
        top_k_movies=args.top_k_movies,
        sample_frac=args.sample_frac,
        seed=args.seed,
    )
    print(f"After filtering rows: {len(filtered):,} | users: {filtered['userId'].nunique():,} | movies: {filtered['movieId'].nunique():,}")

    print("Splitting per user (val = last timestamp)...")
    train_raw, val_raw = per_user_last_timestamp_split(filtered)
    print(f"Train rows: {len(train_raw):,} | Val rows: {len(val_raw):,}")

    # Build indices from train+val together so item_idx is consistent across both files
    combined = pd.concat([train_raw, val_raw], ignore_index=True)
    mapped_all, items_df = build_contiguous_indices(combined)

    # Re-split mapped data by sizes (since we concatenated train_raw then val_raw)
    train_mapped = mapped_all.iloc[: len(train_raw)].reset_index(drop=True)
    val_mapped = mapped_all.iloc[len(train_raw) :].reset_index(drop=True)

    # Save outputs
    train_path = save_csv(train_mapped, os.path.join(args.outdir, "mf_train"))
    val_path = save_csv(val_mapped, os.path.join(args.outdir, "mf_val"))

    items_df = maybe_join_movies_metadata(items_df, args.movies)
    items_path = save_csv(items_df, os.path.join(args.outdir, "mf_items"))

    print("\nWrote:")
    print(f"{train_path}(columns: {list(train_mapped.columns)})")
    print(f"{val_path}  (columns: {list(val_mapped.columns)})")
    print(f"{items_path} columns: {list(items_df.columns)})")

    print("\nNext step (training):")
    print("Load train/val files with columns [user_idx, item_idx, rating]")
    print("num_users = train['user_idx'].nunique(), num_items = items['item_idx'].nunique()")
    print("Train MF embeddings on those indices")


if __name__ == "__main__":
    main()