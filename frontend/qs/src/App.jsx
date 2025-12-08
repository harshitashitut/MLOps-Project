// import { BrowserRouter, Routes, Route } from 'react-router-dom';
// import Layout from './components/Layout';
// import Home from './pages/Home';

// function App() {
//   return (
//     <BrowserRouter>
//       <Layout currentPageName="Home">
//         <Routes>
//           <Route path="/" element={<Home />} />
//         </Routes>
//       </Layout>
//     </BrowserRouter>
//   );
// }

// export default App;

import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import Layout from './components/Layout';
import Home from './pages/Home';
import Login from './pages/Login';
import Signup from './pages/Signup';
import ProtectedRoute from './components/ProtectedRoute';
import PublicSpeaking from './pages/PublicSpeaking';
import InterviewAnalysis from './pages/InterviewAnalysis';
import Pitching from './pages/Pitching';


function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route path="/signup" element={<Signup />} />
        <Route path="/public-speaking" element={<PublicSpeaking />} />
        <Route path="/interview-analysis" element={<InterviewAnalysis />} />
        <Route path="/pitching" element={<Pitching />} />
        <Route
          path="/"
          element={
            <ProtectedRoute>
              <Layout currentPageName="Home">
                <Home />
              </Layout>
            </ProtectedRoute>
          }
        />
      </Routes>
    </BrowserRouter>
  );
}

export default App;