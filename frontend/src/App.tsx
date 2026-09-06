import { BrowserRouter, Routes, Route, Link } from "react-router-dom";
import HomePage from "./pages/HomePage";
import ResultPage from "./pages/ResultPage";
import NotFoundPage from "./pages/NotFoundPage";

export default function App() {
  return (
    <BrowserRouter>
      <div className="app-shell">
        <header className="site-header">
          <div className="site-header__inner">
            <Link to="/" className="wordmark">
              Ke<span>r</span>b
            </Link>
            <span className="site-header__tagline">Property valuations, honestly estimated</span>
          </div>
        </header>

        <main className="site-main">
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/result" element={<ResultPage />} />
            <Route path="*" element={<NotFoundPage />} />
          </Routes>
        </main>

        <footer className="site-footer">
          <div className="site-footer__inner">
            Estimated using a Random Forest model trained on real property listings.
          </div>
        </footer>
      </div>
    </BrowserRouter>
  );
}
