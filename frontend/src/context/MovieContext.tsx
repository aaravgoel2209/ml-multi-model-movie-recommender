import React, { createContext, useContext, useState } from 'react';

// Define the shape of your movie object
interface DroppedMovie {
    id: string;
    title: string;
}

interface MovieContextType {
    favoriteMovies: (DroppedMovie | null)[];
    dislikedMovies: (DroppedMovie | null)[];
    addToFavorites: (movie: DroppedMovie) => void;
    addToDisliked: (movie: DroppedMovie) => void;
    removeMovie: (index: number, section: 'favorite' | 'disliked') => void;
    resetLists: () => void;
}

const MovieContext = createContext<MovieContextType | undefined>(undefined);

export const MovieProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
    const [favoriteMovies, setFavoriteMovies] = useState<(DroppedMovie | null)[]>([null, null, null]);
    const [dislikedMovies, setDislikedMovies] = useState<(DroppedMovie | null)[]>([null, null, null]);

    const addToFavorites = (movie: DroppedMovie) => {
        const index = favoriteMovies.findIndex(m => m === null);
        if (index !== -1) {
            const updated = [...favoriteMovies];
            updated[index] = movie;
            setFavoriteMovies(updated);
        }
    };

    const addToDisliked = (movie: DroppedMovie) => {
        const index = dislikedMovies.findIndex(m => m === null);
        if (index !== -1) {
            const updated = [...dislikedMovies];
            updated[index] = movie;
            setDislikedMovies(updated);
        }
    };

    const removeMovie = (index: number, section: 'favorite' | 'disliked') => {
        if (section === 'favorite') {
            const updated = [...favoriteMovies];
            updated[index] = null;
            setFavoriteMovies(updated);
        } else {
            const updated = [...dislikedMovies];
            updated[index] = null;
            setDislikedMovies(updated);
        }
    };

    const resetLists = () => {
        setFavoriteMovies([null, null, null]);
        setDislikedMovies([null, null, null]);
    };

    return (
        <MovieContext.Provider value={{ favoriteMovies, dislikedMovies, addToFavorites, addToDisliked, removeMovie, resetLists }}>
            {children}
        </MovieContext.Provider>
    );
};

export const useMovies = () => {
    const context = useContext(MovieContext);
    if (!context) throw new Error("useMovies must be used within MovieProvider");
    return context;
};