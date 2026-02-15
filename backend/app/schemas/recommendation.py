from typing import List, Optional, Literal
from pydantic import BaseModel, Field

class RecommendationCandidate(BaseModel):
    """
    Standardized output from any recommendation model (Collaborative, Content, Review).
    
    Addresses:
    - Issue #9: Standardized format (user_id, movie_id, score)
    - FR-4.4: Ranking (score)
    - FR-3.2: Explanations ("Because you liked X")
    - Two Datasets Support: 'dataset_source' to prevent ID collisions.
    """
    # Core Identification
    user_id: str = Field(..., description="ID of the user (string to support multiple ID formats)")
    movie_id: str = Field(..., description="ID of the recommended movie (string to support multiple datasets)")
    dataset_source: Literal["movielens", "rotten_tomatoes"] = Field(
        ..., 
        description="Origin of the movie ID. strictly 'movielens' or 'rotten_tomatoes'"
    )
    
    # Ranking Data
    score: float = Field(..., ge=0.0, le=1.0, description="Normalized confidence score (0.0 to 1.0)")
    model_name: str = Field(..., description="Name of the model source (e.g., 'collaborative_v1')")
    
    # Explanation Data (FR-3.2)
    reason_movie_ids: Optional[List[str]] = Field(
        default=None, 
        description="List of user's past movie IDs that triggered this recommendation"
    )
    reason_genres: Optional[List[str]] = Field(
        default=None, 
        description="List of genres or keywords that matched"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "101",
                "movie_id": "550",
                "dataset_source": "movielens",
                "score": 0.85,
                "model_name": "collaborative_filtering",
                "reason_movie_ids": ["10", "25"],
                "reason_genres": ["Drama", "Thriller"]
            }
        }