"""
Data Links: MovieLens: https://grouplens.org/datasets/movielens/32m/
            Rotten Tomatoes: https://www.kaggle.com/datasets/rotten-tomatoes

Load MovieLens and Rotten Tomatoes datasets into pandas DataFrames.

Usage:
    from data.load_data import load_movielens, load_rotten_tomatoes, train_test_split_ratings
    
    # Load data
    ml_data = load_movielens()
    ratings = ml_data['ratings']
    movies = ml_data['movies']
    
    # Split ratings into train/test
    train, test = train_test_split_ratings(ratings, test_size=0.2)
"""

import pandas as pd
from sklearn.model_selection import train_test_split


def load_movielens(data_dir="data/raw/ml-32m"):
    """Load MovieLens dataset files into DataFrames.
    
    Args:
        data_dir: Path to MovieLens data directory
        
    Returns:
        dict: Dictionary with keys 'ratings', 'movies', 'tags', 'links'
    """
    return {
        'ratings': pd.read_csv(f"{data_dir}/ratings.csv"),
        'movies': pd.read_csv(f"{data_dir}/movies.csv"),
        'tags': pd.read_csv(f"{data_dir}/tags.csv"),
        'links': pd.read_csv(f"{data_dir}/links.csv")
    }


def load_rotten_tomatoes(data_dir="data/raw/rotten-tomato"):
    """Load Rotten Tomatoes dataset files into DataFrames.
    
    Args:
        data_dir: Path to Rotten Tomatoes data directory
        
    Returns:
        dict: Dictionary with keys 'movies', 'reviews'
    """
    return {
        'movies': pd.read_csv(f"{data_dir}/rotten_tomatoes_movies.csv"),
        'reviews': pd.read_csv(f"{data_dir}/rotten_tomatoes_movie_reviews.csv")
    }


def train_test_split_ratings(ratings_df, test_size=0.2, random_state=42):
    """Split ratings DataFrame into train and test sets.
    
    Args:
        ratings_df: DataFrame containing ratings
        test_size: Proportion of data for test set (default: 0.2)
        random_state: Random seed for reproducibility (default: 42)
        
    Returns:
        tuple: (train_df, test_df)
    """
    return train_test_split(ratings_df, test_size=test_size, random_state=random_state)


if __name__ == "__main__":
    print("Loading MovieLens data...")
    ml_data = load_movielens()
    print(f"Ratings: {ml_data['ratings'].shape}")
    print(f"Movies: {ml_data['movies'].shape}")
    print(f"Tags: {ml_data['tags'].shape}")
    print(f"Links: {ml_data['links'].shape}")
    
    print("\nLoading Rotten Tomatoes data...")
    rt_data = load_rotten_tomatoes()
    print(f"Movies: {rt_data['movies'].shape}")
    print(f"Reviews: {rt_data['reviews'].shape}")
    
    print("\nCreating train/test split...")
    train, test = train_test_split_ratings(ml_data['ratings'])
    print(f"Train: {train.shape}")
    print(f"Test: {test.shape}")
    print("\nAll datasets loaded successfully!")
