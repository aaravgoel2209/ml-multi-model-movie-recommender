import React, { useState } from 'react';
import './Nav.css';

export const Nav: React.FC = () => {
    const [isOpen, setIsOpen] = useState(true);
    const [movies, setMovies] = useState(['', '', '']);

    const toggleNav = () => {
        setIsOpen(!isOpen);
    };

    React.useEffect(() => {
        const newWidth = isOpen ? '350px' : '100px';
        document.documentElement.style.setProperty('--nav-width', newWidth);
    }, [isOpen]);

    const handleMovieChange = (index: number, value: string) => {
        const updatedMovies = [...movies];
        updatedMovies[index] = value;
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
                                <input
                                    key={index}
                                    type="text"
                                    className="movie-input"
                                    placeholder={`Movie ${index + 1}`}
                                    value={movie}
                                    onChange={(e) => handleMovieChange(index, e.target.value)}
                                />
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