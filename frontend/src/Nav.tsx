import React, { useState } from 'react';
//import './Nav.css';

export const Nav: React.FC = () => {
    const [isOpen, setIsOpen] = useState(true);

    const toggleNav = () => {
        setIsOpen(!isOpen);
    };

    return (
        <nav className={`navbar ${isOpen ? 'open' : 'closed'}`}>
            {isOpen && <h2 className="navbar-title">MMMR</h2>}
            
            <button
                className="toggle-btn"
                onClick={toggleNav}
                aria-label="Toggle navbar"
            >
                <span className={`arrow ${isOpen ? 'left' : 'right'}`}>▶</span>
            </button>
        </nav>
    );
};