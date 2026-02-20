import { useState, useEffect } from 'react';
import HomeButton from '../../components/HomeButton';
import Card from '../../components/Card';

interface RecommendPageProps {
  onEnter?: () => void;
}

const RECOMMENDED_MOVIES = [
  { id: 'rec-1', title: 'Recommended Movie 1', description: 'This is a recommended movie based on your preferences.' },
  { id: 'rec-2', title: 'Recommended Movie 2', description: 'This is a recommended movie based on your preferences.' },
  { id: 'rec-3', title: 'Recommended Movie 3', description: 'This is a recommended movie based on your preferences.' },
  { id: 'rec-4', title: 'Recommended Movie 4', description: 'This is a recommended movie based on your preferences.' },
  { id: 'rec-5', title: 'Recommended Movie 5', description: 'This is a recommended movie based on your preferences.' },
];

function RecommendPage({ onEnter }: RecommendPageProps) {
  const [colorValue, setColorValue] = useState(360);
  const saturation = colorValue === 360 ? 0 : 50;

  useEffect(() => {
    document.documentElement.style.setProperty('--color_value', colorValue.toString());
    document.documentElement.style.setProperty('--saturation_level', `${saturation}%`);
  }, [colorValue, saturation]);

  const handleDragStart = () => {
    // Handle drag start if needed
  };

  useEffect(() => {
    onEnter?.();
  }, [onEnter]);

  return (
    <>
      <HomeButton />
      <section>
        <div className="recommend-header">
          <h1>Recommendations</h1>
          <input 
            type="range" 
            min="0" 
            max="360" 
            value={colorValue} 
            onChange={(e) => setColorValue(parseInt(e.target.value))}
            className="color-range"
          />
        </div>
        <div>
          <ul className='cards'>
            {RECOMMENDED_MOVIES.map((movie) => (
              <Card
                key={movie.id}
                id={movie.id}
                title={movie.title}
                description={movie.description}
                onDragStart={handleDragStart}
              />
            ))}
          </ul>
        </div>
      </section>
    </>
  )
}

export default RecommendPage
