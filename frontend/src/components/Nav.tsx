import React, { useState, useEffect } from 'react';
import './Nav.css';

import { useMovies } from '../context/MovieContext';

export const Nav: React.FC = () => {
    const [isOpen, setIsOpen] = useState(true);
    const { favoriteMovies, dislikedMovies, addToFavorites, addToDisliked, removeMovie, resetLists } = useMovies();

    const [dragOverIndex, setDragOverIndex] = useState<number | null>(null);
    const [dragOverSection, setDragOverSection] = useState<'favorite' | 'disliked' | null>(null);

    const toggleNav = () => {
        setIsOpen(!isOpen);
    };

    useEffect(() => {
        const newWidth = isOpen ? '350px' : '100px';
        document.documentElement.style.setProperty('--nav-width', newWidth);
    }, [isOpen]);

    const handleDragOver = (e: React.DragEvent<HTMLDivElement>, index: number, section: 'favorite' | 'disliked') => {
        e.preventDefault();
        e.dataTransfer.dropEffect = 'copy';
        setDragOverIndex(index);
        setDragOverSection(section);
    };

    const handleDragLeave = () => {
        setDragOverIndex(null);
        setDragOverSection(null);
    };

    const handleDropFavorite = (e: React.DragEvent<HTMLDivElement>, _index: number) => {
        e.preventDefault();
        handleDragLeave();

        const movieTitle = e.dataTransfer.getData('movieTitle');
        const movieId = e.dataTransfer.getData('movieId');

        if (movieTitle && movieId) {
            addToFavorites({ id: movieId, title: movieTitle });
        }
    };

    const handleDropDisliked = (e: React.DragEvent<HTMLDivElement>, _index: number) => {
        e.preventDefault();
        handleDragLeave();

        const movieTitle = e.dataTransfer.getData('movieTitle');
        const movieId = e.dataTransfer.getData('movieId');

        if (movieTitle && movieId) {
            addToDisliked({ id: movieId, title: movieTitle });
        }
    };


    return (
        <>
            <nav className={`navbar ${isOpen ? 'open' : 'closed'}`}>
                <div className="navbar-header">
                    <h2 className="navbar-title">MMMR</h2>
                    {isOpen && (
                        <button
                            className="restart-btn"
                            onClick={resetLists} 
                            aria-label="Restart and clear all movies"
                            title="Clear all movies"
                        >
                            ↻
                        </button>
                    )}
                </div>

                {isOpen && (
                    <div className="nav-content">
                        <h3 className="nav-subtitle">Insert 3 favorite movies</h3>
                        <div className="movies-input-container favorite-movies">
                            {favoriteMovies.map((movie, index) => (
                                <div
                                    key={index}
                                    className={`movie-drop-zone ${dragOverIndex === index && dragOverSection === 'favorite' ? 'drag-over' : ''}`}
                                    onDragOver={(e) => handleDragOver(e, index, 'favorite')}
                                    onDragLeave={handleDragLeave}
                                    onDrop={(e) => handleDropFavorite(e, index)}
                                >
                                    {movie ? (
                                        <div className="movie-item">
                                            <p className="title">{movie.title}</p>
                                            <button
                                                className="remove-btn"
                                                onClick={() => removeMovie(index, 'favorite')}
                                                aria-label="Remove movie"
                                            >
                                                ✕
                                            </button>
                                        </div>
                                    ) : (
                                        <span className="placeholder">Drop movie here</span>
                                    )}
                                </div>
                            ))}
                        </div>
                        <br />
                        <h3 className="nav-subtitle">Insert 3 movies you dislike</h3>
                        <div className="movies-input-container disliked-movies">
                            {dislikedMovies.map((movie, index) => (
                                <div
                                    key={index}
                                    className={`movie-drop-zone ${dragOverIndex === index && dragOverSection === 'disliked' ? 'drag-over' : ''}`}
                                    onDragOver={(e) => handleDragOver(e, index, 'disliked')}
                                    onDragLeave={handleDragLeave}
                                    onDrop={(e) => handleDropDisliked(e, index)}
                                >
                                    {movie ? (
                                        <div className="movie-item">
                                            <p className="title">{movie.title}</p>
                                            <button
                                                className="remove-btn"
                                                onClick={() => removeMovie(index, 'disliked')}
                                                aria-label="Remove movie"
                                            >
                                                ✕
                                            </button>
                                        </div>
                                    ) : (
                                        <span className="placeholder">Drop movie here</span>
                                    )}
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                <button
                    className="toggle-btn"
                    onClick={toggleNav}
                    aria-label="Toggle navbar"
                >
                    <span className={`arrow ${isOpen ? 'left' : 'right'}`}>◀</span>
                </button>
            </nav>
        </>
    );
};