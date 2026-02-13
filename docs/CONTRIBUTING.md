# How to Contribute v1.0

## Overview

This repository contains a collaborative AI-powered movie recommendation
system.

Key characteristics:

- Multi-model architecture
- Multi-source data ingestion
- Clear separation of data, model, and service layers

All changes must go through Pull Requests.

Direct commits to `main` are prohibited.

---

## General Discipline

- Every change must start from a GitHub Issue
- One issue → one branch → one PR
- Do not mix unrelated changes
- Keep commits atomic and logically consistent
- No experimental or temporary code in `main`
- Code must be reproducible and testable

---

## Workflow

### 1. Issue Assignment

Work begins only after an issue is created and assigned.

Each issue must define:

- Objective
- Scope
- Acceptance criteria

No implementation without an issue.

---

### 2. Branch Creation

Create branch from `main`:

    git checkout main
    git pull
    git checkout -b <branch-name>

---

### 3. Development

- Keep commits small and meaningful
- Follow commit convention
- Ensure tests pass locally

---

### 4. Pull Request

Open a PR targeting `main`.

Requirements:

- Reference issue (`Closes #<id>`)
- CI (Continuous Integration) must pass
- At least one reviewer approval
- No self-merge

---

## Branch Naming Convention

Format:

    <type>/<short-description>

Types:

- `feature/` --- new functionality (API, model, pipeline)
- `fix/` --- bug fix
- `refactor/` --- structural improvement
- `experiment/` --- model or algorithm experiments (non-production)
- `data/` --- data ingestion or preprocessing changes

Rules:

- Lowercase only
- Hyphen-separated words
- Short and descriptive
- No vague names (e.g., `update`, `change`)

Examples:

    feature/hybrid-recommendation
    fix/rating-normalization
    refactor/repository-layer
    experiment/matrix-factorization-v2
    data/imdb-cleaning-pipeline

---

## Commit Message Convention

Format:

    <type>: <short summary>

Allowed types:

- feat
- fix
- refactor
- test
- docs
- chore
- perf

Examples:

    feat: add collaborative filtering model
    fix: correct similarity score calculation
    perf: optimize recommendation ranking query

---

## Pull Request Description Convention

Each PR must follow this structure:

    ## What
    What was implemented.

    ## Why
    Problem being solved.

    ## Scope
    Affected modules (e.g., models/, api/, data_sources/).

    ## How
    High-level implementation approach.

    ## Validation
    - Tests added/updated
    - CI status
    - Metrics (if ML-related)
    - Baseline comparison (if applicable)

    ## Breaking Changes
    Explicitly state if any.

### Additional Requirements for ML PRs

- Report evaluation metrics (e.g., Precision@K, Recall@K, NDCG --- Normalized Discounted Cumulative Gain)
- Specify dataset version used
- Compare against baseline

---

## Protection Rules for `main`

- Direct push disabled
- PR required
- CI must pass
- Minimum 1 approval
- No self-merge

---

This document defines mandatory contribution rules for this repository.
