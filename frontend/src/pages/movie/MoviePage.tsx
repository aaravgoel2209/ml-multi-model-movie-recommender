import { useParams, useNavigate, useLocation } from 'react-router-dom';
import { useEffect } from 'react';
import './MoviePage.css';

interface MoviePageProps {
  onEnter?: () => void;
}

function MoviePage({ onEnter }: MoviePageProps) {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const location = useLocation();

  // Get movie data from URL state or default values
  const state = location.state as { title?: string; description?: string } | null;
  const title = state?.title || 'Movie Title';
  const description = state?.description || 'Movie description not available.';
  // get rating from backend, default value is 5  
  const rating = /* state?.rating || */ 5;

  useEffect(() => {
    onEnter?.();
  }, [onEnter]);

  const handleBack = () => {
    navigate(-1); // Go back to previous page
  };

  return (
    <div className="movie-page">
      <button className="back-btn" onClick={handleBack} aria-label="Go back">
        <svg
          width="24"
          height="24"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <line x1="19" y1="12" x2="5" y2="12" />
          <polyline points="12 19 5 12 12 5" />
        </svg>
        Back
      </button>

      <div className="movie-content">
        <div className="movie-header">
          <h1 className="movie-title">{title}</h1>
          <p className="movie-id">ID: {id}</p>
          <button className="insert-btn" >INSERT</button>
        </div>
        <div className="movie-description">
          <h2>Description</h2>
          <p>{description}</p>
        </div>
        <div className="ratings-container">
          <div className="star-row">
            {[1, 2, 3, 4, 5].map((starNumber) => (
              <span
                key={starNumber}
                style={{ color: starNumber <= rating ? '#ffff00' : '#4b5563' }}
              >
                ★
              </span>
            ))}
          </div>
          <p className="ratings-text">Ratings:</p>
          <div className="dotted-lines">
            <div className="line"></div>
            <div className="line"></div>
            <div className="line"></div>
            <div className="line"></div>
            <div className="line"></div>
            <div className="line"></div>
          </div>
        </div>
        <div className="movie-details">
          <h2>Details</h2>
          <p>Additional movie information will be displayed here when connected to the backend.</p>
        </div>
      </div>
    </div>
  );
}

export default MoviePage;
