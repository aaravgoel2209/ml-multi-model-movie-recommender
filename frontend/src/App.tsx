import { useState, useEffect } from 'react';
import './App.css'

function App() {
  const [colorValue, setColorValue] = useState(360);
  const saturation = colorValue === 360 ? 0 : 50;

  useEffect(() => {
    document.documentElement.style.setProperty('--color_value', colorValue.toString());
    document.documentElement.style.setProperty('--saturation_level', `${saturation}%`);
  }, [colorValue, saturation]);

  return (
    <div>
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
            <li className='card'>
              <h2>Movie Title</h2>
              <p>Movie description goes here. It can be a brief summary of the plot, the genre, or any other relevant information about the movie.</p>
            </li>
            <li className='card'>
              <h2>Movie Title</h2>
              <p>Movie description goes here. It can be a brief summary of the plot, the genre, or any other relevant information about the movie.</p>
            </li>
            <li className='card'>
              <h2>Movie Title</h2>
              <p>Movie description goes here. It can be a brief summary of the plot, the genre, or any other relevant information about the movie.</p>
            </li>
          </ul>
        </div>
      </section>
    </div>
  )
}

export default App
