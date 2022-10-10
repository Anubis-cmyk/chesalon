import './App.css';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';

import HomePage from './pages/HomePage';
import FinishPage from './pages/FinishPage';

function App() {
  return (
     <Router>
          <Routes>
            <Route path="/finish" element={<FinishPage />} />
            <Route path="/" element={<HomePage />} />
          </Routes>
        </Router>
  );
}

export default App;
