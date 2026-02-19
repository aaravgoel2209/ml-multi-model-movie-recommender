# Multi-Model Movie Recommender (MMMR)

## Summary

**What:**  
A movie recommender that suggests films based on movies you already love.

**Why:**  
Finding your next favorite movie is hard. We make it easy.

**How it works (3 steps):**

1. 🎬 You pick 3 movies you already like (or pick from trending suggestions if you're new)
2. 🤖 Our AI analyzes them (looking at plot, genre, keywords, reviews)
3. 🍿 We give you 3 perfect recommendations ranked just for you

**What makes it cool:**  
It uses multiple "mini-AIs" (like Lego blocks) each looking at different things—one reads synopses, another looks at keywords, another checks reviews. Then a final ranking model combines everything.

**Repository:**  
https://github.com/aaravgoel2209/ml-multi-model-movie-recommender

---

# Purpose & Vision

**Problem:**  
Finding a good movie to watch takes too long. Users scroll endlessly through streaming platforms without making a choice.

**Solution:**  
MMMR (Multi-Model Movie Recommender) takes 3 movies a user loves and uses multiple AI models to analyze their plot, genre, and keywords, then returns 3 highly personalized recommendations.

**Vision:**  
Build a modular, Lego-like AI system where different models handle different aspects of movie understanding, all working together to make perfect suggestions. The system should still function even if some models are disabled.

---

# User Stories

- As a movie lover, I want to select 3 movies I enjoy, so that I can discover new films tailored to my taste.
- As a new user with no watch history, I want to pick from trending movies, so that I can still get recommendations without knowing what to search for.
- As a casual viewer, I want to see why a movie was recommended, so that I can decide if it's worth my time.

---

# Scope

## In-Scope

- Web interface for selecting movies
- Search and trending movie grid
- AI that processes 3 or fewer input movies and returns at least 3 recommendations
- Basic explanation for each recommendation
- Clickable cards with detailed movie view
- No login or user accounts required

## Out-of-Scope

- User accounts / profiles / watch history
- Social features (sharing, friends)
- Streaming integration (Netflix links, etc.)
- Mobile apps (web-only first)
- User ratings or feedback loop
- More than 5 recommendations

---

# Functional Requirements

## FR-1: Movie Selection

- **FR-1.1:** User can search for movies by title
- **FR-1.2:** User can browse a grid of trending/popular movies
- **FR-1.3:** User can select up to 3 movies
- **FR-1.4:** Selected movies appear as visual chips with remove option
- **FR-1.5:** "Get Recommendations" button enables ONLY when 3 movies are selected

## FR-2: Movie Display

- **FR-2.1:** Movies appear as cards with poster and title
- **FR-2.2:** Clicking a card opens a detailed view with all available movie data (synopsis, genres, cast, keywords, etc.)
- **FR-2.3:** Detailed view shows as much information as the dataset allows

## FR-3: Recommendations

- **FR-3.1:** System displays 3 movie recommendations as cards
- **FR-3.2:** Each recommendation shows:
  - Poster
  - Title
  - Explanation ("Because you liked [movie]" AND genre/keyword-based reasons — treated equally)
- **FR-3.3:** Recommendations are clickable → same detailed view
- **FR-3.4:** "Try Again" button resets to movie selection

## FR-4: AI Processing

- **FR-4.1:** Backend receives 3 movie IDs
- **FR-4.2:** System fetches synopsis, genre, keywords for each
- **FR-4.3:** Multiple models analyze different aspects
- **FR-4.4:** Final ranking model combines scores
- **FR-4.5:** Returns top 3 results

## FR-5: Data Management

- **FR-5.1:** System loads movie dataset at startup
- **FR-5.2:** Movie dataset contains all available movie-related fields (exact fields TBD)

---

# Non-Functional Requirements

## NFR-1: Performance

- **NFR-1.1:** Search results appear within 300ms
- **NFR-1.2:** Recommendations generated within 20 seconds maximum
- **NFR-1.3:** System handles concurrent users (TBD)

## NFR-2: Usability

- **NFR-2.1:** Interface works on desktop and mobile browsers
- **NFR-2.2:** Clear loading states for all async operations
- **NFR-2.3:** Error messages are user-friendly

## NFR-3: Code Quality

- **NFR-3.1:** TypeScript used throughout frontend
- **NFR-3.2:** Python type hints in backend
- **NFR-3.3:** Testing requirements (TBD)

---

# Constraints & Dependencies

## Constraints

- Frontend: React + TypeScript (Vite)
- Backend: FastAPI (Python)
- AI: PyTorch
- Team: All beginners, learning as we build
- Time: No deadline (learning project)

## Dependencies

- Movie dataset: TBD (TMDB? IMDb?)
- Poster images: Need TMDB API or local fallback
- Hosting/platform: TBD
- Additional APIs: TBD

---

# Project Architecture

```
ml-multi-model-movie-recommender/
├── backend/                # FastAPI (Python) Logic
│   ├── app/
│   │   ├── main.py         # Entry point (initializes FastAPI + CORS)
│   │   ├── api/            # API endpoints (POST /recommend, GET /search, GET /trending)
│   │   ├── services/       # Core business logic
│   │   │   └── ai_engine.py # Final ranking logic (calls LEGO blocks)
│   │   ├── schemas/        # Pydantic models (data validation)
│   │   └── utils/          # Helpers
│   └── requirements.txt    # Python deps
│
├── frontend/               # React + TypeScript (Vite) UI
│   ├── src/
│   │   ├── api/
│   │   ├── components/
│   │   ├── hooks/
│   │   ├── types/
│   │   ├── pages/
│   │   └── App.tsx
│   └── package.json
│
├── data/
├── docs/
├── evaluation/
└── models/
```

---

# Team Responsibilities (GitHub Issues)

## Phase 1: Foundation

- #2 Select and Document Dataset
- #3 Implement Data Loading Script
- #4 Implement Basic Data Cleaning
- #5 Create Train/Test Split

## Phase 2: Individual Models (Lego Blocks)

- #6 Build Collaborative Filtering Model
- #7 Build Metadata-Based Recommender
- #8 Build Review-Based Text Recommender

## Phase 3: Multi-Model System

- #9 Standardize Candidate Output Format
- #10 Combine Candidate Outputs
- #11 Create Feature Table for Ranking
- #12 Implement Final Ranking Model
- #14 Compare Individual Models vs Final Model

## Phase 4: Application

- #13 Implement Evaluation Metrics
- #15 Build Backend API Endpoint
- #16 Create Simple Frontend Interface
- #17 Create System Architecture Diagram
