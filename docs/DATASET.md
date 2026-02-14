# Data Documentation

## Overview

This project uses two primary datasets to support a multi-model movie recommendation system:

1. **MovieLens 32M** — User–movie interaction dataset used for collaborative filtering.
2. **Massive Rotten Tomatoes Dataset (Kaggle)** — Movie metadata and critic reviews used for content-based and review-based modeling.

The `raw/` directory contains original downloaded data files.  
The `cleaned/` directory contains processed datasets used for modeling and experimentation.

Raw datasets are excluded from version control due to size limitations. They must be downloaded separately from their official sources.

---

# 1. MovieLens 32M Dataset

**Source:**  
https://grouplens.org/datasets/movielens/32m/

## Dataset Summary

- ~32,000,000 ratings  
- 200,948 users  
- 84,432 movies  
- Rating scale: 0.5 – 5.0 (half-star increments)  
- Minimum 20 ratings per user  

This dataset forms the foundation for collaborative filtering models.

---

## Files Included

### ratings.csv

Each row represents one user rating one movie.

| Column     | Type    | Description |
|------------|---------|-------------|
| userId     | int64   | Unique anonymized user identifier |
| movieId    | int64   | Unique movie identifier |
| rating     | float64 | Rating given by user (0.5–5.0) |
| timestamp  | int64   | Unix timestamp (seconds since Jan 1, 1970 UTC) |

Notes:
- Ratings are ordered by `userId`, then by `movieId`.
- Timestamps are stored in Unix time format.

---

### movies.csv

Movie-level metadata.

| Column   | Type   | Description |
|----------|--------|-------------|
| movieId  | int64  | Unique movie identifier |
| title    | object | Movie title (includes release year) |
| genres   | object | Pipe-separated list of genres |

Genres are multi-label and drawn from predefined categories (e.g., Action, Drama, Comedy, Sci-Fi).

---

### tags.csv

User-generated descriptive tags applied to movies.

| Column    | Type   | Description |
|-----------|--------|-------------|
| userId    | int64  | User identifier |
| movieId   | int64  | Movie identifier |
| tag       | object | User-provided tag |
| timestamp | int64  | Unix timestamp |

Tags provide additional contextual metadata for hybrid and content-based models.

---

### links.csv

External identifiers for integration with third-party data sources.

| Column  | Type    | Description |
|---------|---------|-------------|
| movieId | int64   | MovieLens movie ID |
| imdbId  | int64   | IMDb identifier |
| tmdbId  | float64 | TMDB identifier |

These identifiers enable enrichment via external APIs (e.g., posters, extended metadata).

---

## Data Characteristics

- The user–item interaction matrix is highly sparse (~99.8%).
- Movie popularity follows a long-tail distribution:
  - Many movies have very few ratings.
  - A small subset of movies receive very high engagement.
- Users are moderately active due to the minimum 20-rating threshold.

These properties make the dataset suitable for matrix factorization and large-scale collaborative filtering methods.

---

# 2. Massive Rotten Tomatoes Dataset

**Source:**  
https://www.kaggle.com/datasets/andrezaza/clapper-massive-rotten-tomatoes-movies-and-reviews

This dataset provides movie-level metadata and critic review text, enabling sentiment analysis and hybrid recommendation modeling.

---

## Dataset Summary

### Movies File
- 143,258 movies
- Contains audience scores, critic scores, and rich metadata fields

### Reviews File
- 1,444,963 critic reviews
- Includes full review text and sentiment labels

---

## Files Included

### rotten_tomatoes_movies.csv

Movie-level metadata.

| Column | Description |
|--------|-------------|
| id | Unique movie identifier |
| title | Movie title |
| audienceScore | Audience score (percentage) |
| tomatoMeter | Critic score |
| rating | Content rating (e.g., PG-13) |
| ratingContents | Rating explanation |
| releaseDateTheaters | Theatrical release date |
| releaseDateStreaming | Streaming release date |
| runtimeMinutes | Duration in minutes |
| genre | Comma-separated genres |
| originalLanguage | Original language |
| director | Director(s) |
| writer | Writer(s) |
| boxOffice | Box office earnings |
| distributor | Distributor |
| soundMix | Audio format |

Several financial and critic-related columns contain missing values.

---

### rotten_tomatoes_movie_reviews.csv

Critic review dataset.

| Column | Description |
|--------|-------------|
| id | Movie identifier (joins with movies file) |
| reviewId | Unique review identifier |
| creationDate | Review publication date |
| criticName | Name of critic |
| isTopCritic | Boolean indicator |
| originalScore | Score assigned by critic |
| reviewState | Fresh / Rotten |
| publicatioName | Publication name |
| reviewText | Full review content |
| scoreSentiment | POSITIVE / NEGATIVE |
| reviewUrl | Link to original review |

This dataset supports:
- Sentiment classification
- Text embedding models
- Review-based ranking signals
- Hybrid recommendation approaches

---

## Data Characteristics

- Large-scale text dataset (~1.4M reviews).
- Includes both structured sentiment labels and unstructured text.
- Movie metadata contains partial missing values in box office and critic metrics.
- Review sentiment labels enable supervised NLP modeling.

---


