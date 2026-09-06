import { useState } from "react";
import { useNavigate } from "react-router-dom";
import PredictionForm from "../components/PredictionForm";
import { predictPrice, ApiError } from "../api/predictionClient";
import type { PredictionRequest } from "../types/prediction";

export default function HomePage() {
  const navigate = useNavigate();
  const [isLoading, setIsLoading] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  async function handleSubmit(payload: PredictionRequest) {
    setIsLoading(true);
    setSubmitError(null);
    try {
      const result = await predictPrice(payload);
      navigate("/result", { state: { request: payload, response: result } });
    } catch (err) {
      if (err instanceof ApiError) {
        setSubmitError(
          typeof err.detail === "string" ? err.detail : "Something went wrong. Try again."
        );
      } else {
        setSubmitError("Something went wrong. Try again.");
      }
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="home-grid">
      <div className="home-intro">
        <h1>
          What is your property actually worth?
        </h1>
        <p>
          Kerb estimates real-estate value from a model trained on 139,619
          real listings across India — no guesswork, no synthetic data.
          Fill in the details on the right for a considered estimate.
        </p>
        <div className="home-intro__stats">
          <div className="stat-line">
            <strong>50</strong>
            <span>real cities covered, from the training data itself</span>
          </div>
          <div className="stat-line">
            <strong>₹1.10L</strong>
            <span>typical estimate error (median), not hidden behind a vague score</span>
          </div>
        </div>
      </div>

      <PredictionForm
        onSubmit={handleSubmit}
        isLoading={isLoading}
        submitError={submitError}
      />
    </div>
  );
}
