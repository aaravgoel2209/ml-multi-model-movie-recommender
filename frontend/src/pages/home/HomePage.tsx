import { useState, useEffect } from 'react';
import Card from '../../components/Card';

interface Movie {
    id: string;
    title: string;
    description: string;
}

const MOVIES: Movie[] = [
    { id: 'movie-1', title: 'The Shawshank Redemption', description: 'Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.' },
    { id: 'movie-2', title: 'The Godfather', description: 'The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his youngest son.' },
    { id: 'movie-3', title: 'The Dark Knight', description: 'When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological tests.' },
    { id: 'movie-4', title: 'Pulp Fiction', description: 'The lives of two mob hitmen, a boxer, a gangster and his wife intertwine in four tales of violence and redemption.' },
    { id: 'movie-5', title: 'Forrest Gump', description: 'The presidencies of Kennedy and Johnson unfold from the perspective of an Alabama man with an IQ of 75.' },
];

function HomePage() {
  const [colorValue, setColorValue] = useState(360);
  const saturation = colorValue === 360 ? 0 : 50;
  const [draggedMovie, setDraggedMovie] = useState<{ id: string; title: string } | null>(null);

  useEffect(() => {
    document.documentElement.style.setProperty('--color_value', colorValue.toString());
    document.documentElement.style.setProperty('--saturation_level', `${saturation}%`);
  }, [colorValue, saturation]);

  const handleDragStart = (e: React.DragEvent<HTMLLIElement>, id: string, title: string) => {
    setDraggedMovie({ id, title });
    e.dataTransfer.effectAllowed = 'copy';
  };

  if (draggedMovie && !draggedMovie.id) {
    console.log(draggedMovie);
  }

  return (
    <section>
      <div id="search">
        <h1>Search Movies</h1>
        <input type="text" placeholder="Type a title..." />
        <div>
          <button>Magic Trick</button>
          <input type="range" min="0" max="360" value={colorValue} onChange={(e) => setColorValue(parseInt(e.target.value))}/>
        </div>
      </div>
      <div>
        <ul className='cards'>
          {MOVIES.map((movie) => (
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
)
}

export default HomePage
