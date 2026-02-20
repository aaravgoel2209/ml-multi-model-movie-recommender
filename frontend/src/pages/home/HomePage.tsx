import { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import Card from '../../components/Card';

interface HomePageProps {
  onMagicTrick: () => void;
}

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
    { id: 'movie-6', title: 'Inception', description: 'A skilled thief who steals corporate secrets through the use of dream-sharing technology is given the inverse task of planting an idea.' },
    { id: 'movie-7', title: 'Interstellar', description: 'A team of explorers travel through a wormhole in space in an attempt to ensure humanity\'s survival.' },
    { id: 'movie-8', title: 'The Matrix', description: 'A computer hacker learns from mysterious rebels about the true nature of his reality and his role in the war against its controllers.' },
    { id: 'movie-9', title: 'Gladiator', description: 'A former Roman General sets out to exact vengeance against the corrupt emperor who murdered his family and sent him into slavery.' },
    { id: 'movie-10', title: 'The Silence of the Lambs', description: 'A young FBI cadet must receive the help of an incarcerated cannibal killer to help catch another serial killer who skins his victims.' },
];

function HomePage({ onMagicTrick }: HomePageProps) {
  const navigate = useNavigate();
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

  const handleMagicTrick = useCallback(() => {
    onMagicTrick();
    setTimeout(() => {
      navigate('/recommend');
    }, 500);
  }, [navigate, onMagicTrick]);

  if (draggedMovie && !draggedMovie.id) {
    console.log(draggedMovie);
  }

  return (
    <section>
      <div id="search">
        <h1>Search Movies</h1>
        <input type="text" placeholder="Type a title..." />
        <div>
          <button onClick={handleMagicTrick}>Magic Trick</button>
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
