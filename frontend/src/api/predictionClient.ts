import type { PredictionRequest, PredictionResponse } from "../types/prediction";

// The backend base URL is NEVER hardcoded here -- it comes exclusively
// from the VITE_API_BASE_URL environment variable (see .env / .env.example).
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL;

if (!API_BASE_URL) {
  // Fail loudly during development rather than silently calling a wrong
  // relative URL.
  // eslint-disable-next-line no-console
  console.error(
    "VITE_API_BASE_URL is not set. Create a .env file (see .env.example)."
  );
}

export class ApiError extends Error {
  status: number;
  detail: unknown;

  constructor(status: number, detail: unknown) {
    super(
      typeof detail === "string" ? detail : "The valuation service returned an error."
    );
    this.status = status;
    this.detail = detail;
  }
}

export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE_URL}/health`);
    if (!res.ok) return false;
    const body = await res.json();
    return Boolean(body.model_loaded);
  } catch {
    return false;
  }
}

export async function predictPrice(
  payload: PredictionRequest
): Promise<PredictionResponse> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  } catch {
    throw new ApiError(0, "Could not reach the valuation service. Is the backend running?");
  }

  if (!res.ok) {
    let detail: unknown = `Request failed with status ${res.status}`;
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
    } catch {
      // response wasn't JSON -- keep the generic detail above
    }
    throw new ApiError(res.status, detail);
  }

  return (await res.json()) as PredictionResponse;
}
