import { Suspense, lazy } from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import TopNav from "./components/TopNav.jsx";
import Home from "./pages/Home.jsx";
import Review from "./pages/Review.jsx";
import About from "./pages/About.jsx";

// Graph Studio carries Cytoscape (most of the bundle); load it only when it is opened.
const GraphStudio = lazy(() => import("./pages/GraphStudio.jsx"));

export default function App() {
  return (
    <div className="min-h-screen flex flex-col">
      <TopNav />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/review" element={<Review />} />
        <Route path="/about" element={<About />} />
        <Route
          path="/graph"
          element={
            <Suspense fallback={<p className="text-center text-muted text-sm mt-10">Loading Graph Studio…</p>}>
              <GraphStudio />
            </Suspense>
          }
        />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </div>
  );
}
