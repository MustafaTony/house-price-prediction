import { Navigate, Link, useLocation } from "react-router-dom";
import type { PredictionRequest, PredictionResponse } from "../types/prediction";

interface ResultState {
  request: PredictionRequest;
  response: PredictionResponse;
}

function formatLocationLabel(loc: string): string {
  return loc
    .split("-")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

export default function ResultPage() {
  const location = useLocation();
  const state = location.state as ResultState | null;

  if (!state) {
    // No result to show (e.g. direct navigation or a page refresh) --
    // send back to the form rather than showing a broken page.
    return <Navigate to="/" replace />;
  }

  const { request, response } = state;

  return (
    <div className="result">
      <div className="result__label">Estimated value</div>
      <div className="result__price">{response.predicted_price_formatted}</div>

      <p className="result__recap">
        For a <strong>{request.carpet_area_sqft.toLocaleString()} sq ft</strong>{" "}
        {request.furnishing.toLowerCase()} property in{" "}
        <strong>{formatLocationLabel(request.location)}</strong>, floor{" "}
        {request.floor}, with {request.bathrooms} bathroom
        {request.bathrooms === 1 ? "" : "s"} and {request.balconies} balcon
        {request.balconies === 1 ? "y" : "ies"}, facing {request.facing.toLowerCase()}.
      </p>

      {!response.location_recognized && (
        <div className="result__unrecognized">
          "{formatLocationLabel(request.location)}" wasn't one of the cities in
          the training data, so this estimate leans more heavily on the
          property's other features. Treat it as a rougher guide.
        </div>
      )}

      <div className="result__note">
        <strong>How to read this number.</strong> On real held-out listings,
        this model's typical error was about ₹1.10 lakh, though prices in this
        market range from roughly ₹1 lakh to several hundred crore, so any
        single estimate is a starting point for a conversation, not an
        appraisal.
      </div>

      <div className="result__actions">
        <Link to="/" className="btn-secondary">
          New estimate
        </Link>
      </div>
    </div>
  );
}
