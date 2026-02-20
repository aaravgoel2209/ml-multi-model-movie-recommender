import './App.css'
import { useState } from 'react';
import { Routes, Route } from 'react-router-dom';
import { Nav } from '../components/Nav';
import HomePage from './home/HomePage';
import RecommendPage from './recommend/RecommendPage';
import Loading from '../components/Loading';

function App() {
  const [isLoading, setIsLoading] = useState(false);

  const handleMagicTrick = () => {
    // Set loading state - will remain until backend responds
    setIsLoading(true);
  };

  const handleNavigateComplete = () => {
    // Called when backend data is loaded and RecommendPage is ready
    // This will be triggered by RecommendPage's onEnter callback
    setIsLoading(false);
  };

  return (
    <>
      {/* Always render Routes so navigation works */}
      <div className="App">
        <Nav />
        <Routes>
          <Route path="/" element={<HomePage onMagicTrick={handleMagicTrick} />} />
          <Route path="/recommend" element={<RecommendPage onEnter={handleNavigateComplete} />} />
        </Routes>
      </div>
      
      {/* Loading overlay appears on top of Routes */}
      {isLoading && <Loading />}
    </>
  )
}

export default App
