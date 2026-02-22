import React, { useState, useEffect } from 'react';
import './Nav.css';

interface DroppedMovie {
    id: string;
    title: string;
}

export const Nav: React.FC = () => {
    const [isOpen, setIsOpen] = useState(true);
    const [favoriteMovies, setFavoriteMovies] = useState<(DroppedMovie | null)[]>([null, null, null]);
    const [dislikedMovies, setDislikedMovies] = useState<(DroppedMovie | null)[]>([null, null, null]);
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

    const handleDropFavorite = (e: React.DragEvent<HTMLDivElement>, index: number) => {
        e.preventDefault();
        setDragOverIndex(null);
        setDragOverSection(null);
        
        // Get the movie data from the drag event
        const movieTitle = e.dataTransfer.getData('movieTitle');
        const movieId = e.dataTransfer.getData('movieId');
        
        if (movieTitle && movieId) {
            const updatedMovies = [...favoriteMovies];
            updatedMovies[index] = { id: movieId, title: movieTitle };
            setFavoriteMovies(updatedMovies);
        }
    };

    const handleDropDisliked = (e: React.DragEvent<HTMLDivElement>, index: number) => {
        e.preventDefault();
        setDragOverIndex(null);
        setDragOverSection(null);
        
        // Get the movie data from the drag event
        const movieTitle = e.dataTransfer.getData('movieTitle');
        const movieId = e.dataTransfer.getData('movieId');
        
        if (movieTitle && movieId) {
            const updatedMovies = [...dislikedMovies];
            updatedMovies[index] = { id: movieId, title: movieTitle };
            setDislikedMovies(updatedMovies);
        }
    };

    const removeMovie = (index: number, section: 'favorite' | 'disliked') => {
        if (section === 'favorite') {
            const updatedMovies = [...favoriteMovies];
            updatedMovies[index] = null;
            setFavoriteMovies(updatedMovies);
        } else {
            const updatedMovies = [...dislikedMovies];
            updatedMovies[index] = null;
            setDislikedMovies(updatedMovies);
        }
    };

    const handleRestart = () => {
        setFavoriteMovies([null, null, null]);
        setDislikedMovies([null, null, null]);
    };

    return (
        <>
            <nav 
                className={`navbar ${isOpen ? 'open' : 'closed'}`}
            >
                <div className="navbar-header">
                    <h2 className="navbar-title">MMMR</h2>
                    {isOpen && (
                        <button
                            className="restart-btn"
                            onClick={handleRestart}
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