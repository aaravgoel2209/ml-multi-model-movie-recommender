import React from 'react';
import { useNavigate } from 'react-router-dom';

interface MovieCardProps {
    id: string;
    title: string;
    description: string;
    onDragStart: (e: React.DragEvent<HTMLLIElement>, id: string, title: string) => void;
}

export default function Card({ id, title, description, onDragStart }: MovieCardProps) {
    const navigate = useNavigate();
    
    const handleDragStart = (e: React.DragEvent<HTMLLIElement>) => {
        e.dataTransfer.effectAllowed = 'copy';
        e.dataTransfer.setData('movieId', id);
        e.dataTransfer.setData('movieTitle', title);
        onDragStart(e, id, title);
    };

    const handleClick = () => {
        // Navigate to movie page with movie data
        navigate(`/movie/${id}`, { state: { title, description } });
    };

    return (
        <li 
            className='card'
            draggable
            onDragStart={handleDragStart}
            onClick={handleClick}
        >
            <h2>{title}</h2>
            <p>{description}</p>
        </li>
    );
}