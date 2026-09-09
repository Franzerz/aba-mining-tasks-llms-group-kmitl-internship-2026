import { Navigate, Route, Routes } from "react-router-dom";
import TopNav from "./components/TopNav.jsx";
import Home from "./pages/Home.jsx";
import Review from "./pages/Review.jsx";
import GraphStudio from "./pages/GraphStudio.jsx";

export default function App() {
  return (
    <div className="min-h-screen flex flex-col">
      <TopNav />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/review" element={<Review />} />
        <Route path="/graph" element={<GraphStudio />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </div>
  );
}
