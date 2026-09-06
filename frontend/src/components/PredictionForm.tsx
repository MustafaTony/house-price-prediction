import { useState, type FormEvent } from "react";
import {
  FACING_OPTIONS,
  FURNISHING_OPTIONS,
  OWNERSHIP_OPTIONS,
  TRANSACTION_OPTIONS,
  type PredictionRequest,
} from "../types/prediction";
import locationsData from "../data/locations.json";

const LOCATIONS: string[] = locationsData as string[];

interface FormState {
  location: string;
  carpet_area_sqft: string;
  floor: string;
  bathrooms: string;
  balconies: string;
  furnishing: string;
  transaction: string;
  ownership: string;
  facing: string;
}

const initialState: FormState = {
  location: "",
  carpet_area_sqft: "",
  floor: "",
  bathrooms: "",
  balconies: "",
  furnishing: "",
  transaction: "",
  ownership: "",
  facing: "",
};

type FieldErrors = Partial<Record<keyof FormState, string>>;

interface PredictionFormProps {
  onSubmit: (payload: PredictionRequest) => void;
  isLoading: boolean;
  submitError: string | null;
}

function formatLocationLabel(loc: string): string {
  return loc
    .split("-")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

export default function PredictionForm({
  onSubmit,
  isLoading,
  submitError,
}: PredictionFormProps) {
  const [values, setValues] = useState<FormState>(initialState);
  const [errors, setErrors] = useState<FieldErrors>({});

  function update<K extends keyof FormState>(key: K, value: string) {
    setValues((prev) => ({ ...prev, [key]: value }));
  }

  function validate(): FieldErrors {
    const next: FieldErrors = {};

    if (!values.location.trim()) {
      next.location = "Select the property's location.";
    }

    const area = Number(values.carpet_area_sqft);
    if (!values.carpet_area_sqft) {
      next.carpet_area_sqft = "Enter the carpet area.";
    } else if (Number.isNaN(area) || area <= 0) {
      next.carpet_area_sqft = "Carpet area must be greater than 0.";
    }

    if (values.floor === "" || Number.isNaN(Number(values.floor))) {
      next.floor = "Enter the floor number (0 for ground).";
    }

    if (values.bathrooms === "" || Number(values.bathrooms) < 0) {
      next.bathrooms = "Enter the number of bathrooms.";
    }

    if (values.balconies === "" || Number(values.balconies) < 0) {
      next.balconies = "Enter the number of balconies.";
    }

    if (!values.furnishing) next.furnishing = "Select a furnishing status.";
    if (!values.transaction) next.transaction = "Select a transaction type.";
    if (!values.ownership) next.ownership = "Select an ownership type.";
    if (!values.facing) next.facing = "Select which way the property faces.";

    return next;
  }

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    const validationErrors = validate();
    setErrors(validationErrors);
    if (Object.keys(validationErrors).length > 0) return;

    onSubmit({
      location: values.location,
      carpet_area_sqft: Number(values.carpet_area_sqft),
      floor: Number(values.floor),
      bathrooms: Number(values.bathrooms),
      balconies: Number(values.balconies),
      furnishing: values.furnishing as PredictionRequest["furnishing"],
      transaction: values.transaction as PredictionRequest["transaction"],
      ownership: values.ownership as PredictionRequest["ownership"],
      facing: values.facing as PredictionRequest["facing"],
    });
  }

  return (
    <form className="ledger" onSubmit={handleSubmit} noValidate>
      <div className="ledger__heading">Describe the property</div>
      <div className="ledger__subheading">
        Every field below reflects something the model was actually trained
        on — nothing here is decorative.
      </div>

      <div className="field-grid">
        <div className={`field field--full ${errors.location ? "field--invalid" : ""}`}>
          <label htmlFor="location">Location</label>
          <select
            id="location"
            value={values.location}
            onChange={(e) => update("location", e.target.value)}
          >
            <option value="">Select a city</option>
            {LOCATIONS.map((loc) => (
              <option key={loc} value={loc}>
                {formatLocationLabel(loc)}
              </option>
            ))}
          </select>
          {errors.location && <span className="field__error">{errors.location}</span>}
        </div>

        <div className={`field ${errors.carpet_area_sqft ? "field--invalid" : ""}`}>
          <label htmlFor="carpet_area_sqft">Carpet area (sq ft)</label>
          <input
            id="carpet_area_sqft"
            type="number"
            min="1"
            step="1"
            placeholder="e.g. 1200"
            value={values.carpet_area_sqft}
            onChange={(e) => update("carpet_area_sqft", e.target.value)}
          />
          {errors.carpet_area_sqft && (
            <span className="field__error">{errors.carpet_area_sqft}</span>
          )}
        </div>

        <div className={`field ${errors.floor ? "field--invalid" : ""}`}>
          <label htmlFor="floor">Floor</label>
          <input
            id="floor"
            type="number"
            step="1"
            placeholder="0 = ground floor"
            value={values.floor}
            onChange={(e) => update("floor", e.target.value)}
          />
          {errors.floor && <span className="field__error">{errors.floor}</span>}
        </div>

        <div className={`field ${errors.bathrooms ? "field--invalid" : ""}`}>
          <label htmlFor="bathrooms">Bathrooms</label>
          <input
            id="bathrooms"
            type="number"
            min="0"
            step="1"
            value={values.bathrooms}
            onChange={(e) => update("bathrooms", e.target.value)}
          />
          {errors.bathrooms && <span className="field__error">{errors.bathrooms}</span>}
        </div>

        <div className={`field ${errors.balconies ? "field--invalid" : ""}`}>
          <label htmlFor="balconies">Balconies</label>
          <input
            id="balconies"
            type="number"
            min="0"
            step="1"
            value={values.balconies}
            onChange={(e) => update("balconies", e.target.value)}
          />
          {errors.balconies && <span className="field__error">{errors.balconies}</span>}
        </div>

        <div className={`field ${errors.furnishing ? "field--invalid" : ""}`}>
          <label htmlFor="furnishing">Furnishing</label>
          <select
            id="furnishing"
            value={values.furnishing}
            onChange={(e) => update("furnishing", e.target.value)}
          >
            <option value="">Select</option>
            {FURNISHING_OPTIONS.map((opt) => (
              <option key={opt} value={opt}>
                {opt}
              </option>
            ))}
          </select>
          {errors.furnishing && <span className="field__error">{errors.furnishing}</span>}
        </div>

        <div className={`field ${errors.transaction ? "field--invalid" : ""}`}>
          <label htmlFor="transaction">Transaction type</label>
          <select
            id="transaction"
            value={values.transaction}
            onChange={(e) => update("transaction", e.target.value)}
          >
            <option value="">Select</option>
            {TRANSACTION_OPTIONS.map((opt) => (
              <option key={opt} value={opt}>
                {opt}
              </option>
            ))}
          </select>
          {errors.transaction && <span className="field__error">{errors.transaction}</span>}
        </div>

        <div className={`field ${errors.ownership ? "field--invalid" : ""}`}>
          <label htmlFor="ownership">Ownership</label>
          <select
            id="ownership"
            value={values.ownership}
            onChange={(e) => update("ownership", e.target.value)}
          >
            <option value="">Select</option>
            {OWNERSHIP_OPTIONS.map((opt) => (
              <option key={opt} value={opt}>
                {opt}
              </option>
            ))}
          </select>
          {errors.ownership && <span className="field__error">{errors.ownership}</span>}
        </div>

        <div className={`field ${errors.facing ? "field--invalid" : ""}`}>
          <label htmlFor="facing">Facing</label>
          <select
            id="facing"
            value={values.facing}
            onChange={(e) => update("facing", e.target.value)}
          >
            <option value="">Select</option>
            {FACING_OPTIONS.map((opt) => (
              <option key={opt} value={opt}>
                {opt}
              </option>
            ))}
          </select>
          {errors.facing && <span className="field__error">{errors.facing}</span>}
        </div>
      </div>

      <div className="ledger__submit">
        <button type="submit" className="btn-primary" disabled={isLoading}>
          {isLoading ? "Estimating…" : "Estimate value"}
        </button>
        {submitError && <span className="form-status form-status--error">{submitError}</span>}
      </div>
    </form>
  );
}
