import React, { useState, useEffect } from 'react';
import './Nav.css';

interface DroppedMovie {
    id: string;
    title: string;
}

export const Nav: React.FC = () => {
    const [isOpen, setIsOpen] = useState(true);
    const [movies, setMovies] = useState<(DroppedMovie | null)[]>([null, null, null]);
    const [dragOverIndex, setDragOverIndex] = useState<number | null>(null);

    const toggleNav = () => {
        setIsOpen(!isOpen);
    };

    useEffect(() => {
        const newWidth = isOpen ? '350px' : '100px';
        document.documentElement.style.setProperty('--nav-width', newWidth);
    }, [isOpen]);

    const handleDragOver = (e: React.DragEvent<HTMLDivElement>, index: number) => {
        e.preventDefault();
        e.dataTransfer.dropEffect = 'copy';
        setDragOverIndex(index);
    };

    const handleDragLeave = () => {
        setDragOverIndex(null);
    };

    const handleDrop = (e: React.DragEvent<HTMLDivElement>, index: number) => {
        e.preventDefault();
        setDragOverIndex(null);
        
        // Get the movie data from the drag event
        const movieTitle = e.dataTransfer.getData('movieTitle');
        const movieId = e.dataTransfer.getData('movieId');
        
        if (movieTitle && movieId) {
            const updatedMovies = [...movies];
            updatedMovies[index] = { id: movieId, title: movieTitle };
            setMovies(updatedMovies);
        }
    };

    const removeMovie = (index: number) => {
        const updatedMovies = [...movies];
        updatedMovies[index] = null;
        setMovies(updatedMovies);
    };

    return (
        <>
            <nav 
                className={`navbar ${isOpen ? 'open' : 'closed'}`}
            >
                <h2 className="navbar-title">MMMR</h2>
                
                {isOpen && (
                    <div className="nav-content">
                        <h3 className="nav-subtitle">Insert 3 favorite movies</h3>
                        <div className="movies-input-container">
                            {movies.map((movie, index) => (
                                <div
                                    key={index}
                                    className={`movie-drop-zone ${dragOverIndex === index ? 'drag-over' : ''}`}
                                    onDragOver={(e) => handleDragOver(e, index)}
                                    onDragLeave={handleDragLeave}
                                    onDrop={(e) => handleDrop(e, index)}
                                >
                                    {movie ? (
                                        <div className="movie-item">
                                            <span className="movie-title">{movie.title}</span>
                                            <button
                                                className="remove-btn"
                                                onClick={() => removeMovie(index)}
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
                        <div className="movies-input-container">
                            {movies.map((movie, index) => (
                                <div
                                    key={index}
                                    className={`movie-drop-zone ${dragOverIndex === index ? 'drag-over' : ''}`}
                                    onDragOver={(e) => handleDragOver(e, index)}
                                    onDragLeave={handleDragLeave}
                                    onDrop={(e) => handleDrop(e, index)}
                                >
                                    {movie ? (
                                        <div className="movie-item">
                                            <span className="movie-title">{movie.title}</span>
                                            <button
                                                className="remove-btn"
                                                onClick={() => removeMovie(index)}
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