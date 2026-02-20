import { useState, useEffect } from 'react';
import Card from '../../components/Card';

function HomePage() {
  const [colorValue, setColorValue] = useState(360);
  const saturation = colorValue === 360 ? 0 : 50;

  useEffect(() => {
    document.documentElement.style.setProperty('--color_value', colorValue.toString());
    document.documentElement.style.setProperty('--saturation_level', `${saturation}%`);
  }, [colorValue, saturation]);

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
          <Card />
          <Card />
          <Card />
          <Card />
          <Card />
        </ul>
      </div>
    </section>
  )
}

export default HomePage
