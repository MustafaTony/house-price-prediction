import { Link } from "react-router-dom";

export default function NotFoundPage() {
  return (
    <div className="not-found">
      <div className="not-found__code">404</div>
      <p>This page doesn't exist. Estimates live at the valuation form.</p>
      <div className="not-found__actions">
        <Link to="/" className="btn-secondary">
          Back to the form
        </Link>
      </div>
    </div>
  );
}
