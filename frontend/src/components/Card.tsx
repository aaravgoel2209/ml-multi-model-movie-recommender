import React from 'react';

interface MovieCardProps {
    id: string;
    title: string;
    description: string;
    onDragStart: (e: React.DragEvent<HTMLLIElement>, id: string, title: string) => void;
}

export default function Card({ id, title, description, onDragStart }: MovieCardProps) {
    const handleDragStart = (e: React.DragEvent<HTMLLIElement>) => {
        e.dataTransfer.effectAllowed = 'copy';
        e.dataTransfer.setData('movieId', id);
        e.dataTransfer.setData('movieTitle', title);
        onDragStart(e, id, title);
    };

    return (
        <li 
            className='card'
            draggable
            onDragStart={handleDragStart}
        >
            <h2>{title}</h2>
            <p>{description}</p>
        </li>
    );
}