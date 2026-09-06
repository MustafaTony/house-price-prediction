/**
 * Mirrors backend/app/schemas/prediction.py exactly. If the backend
 * schema changes, this file must change with it.
 */

export type Furnishing = "Furnished" | "Semi-Furnished" | "Unfurnished";

export type TransactionType = "New Property" | "Resale" | "Rent/Lease" | "Other";

export type Ownership =
  | "Freehold"
  | "Leasehold"
  | "Co-operative Society"
  | "Power Of Attorney";

export type Facing =
  | "East"
  | "North"
  | "North - East"
  | "North - West"
  | "South"
  | "South - East"
  | "South -West"
  | "West";

export interface PredictionRequest {
  location: string;
  carpet_area_sqft: number;
  floor: number;
  bathrooms: number;
  balconies: number;
  furnishing: Furnishing;
  transaction: TransactionType;
  ownership: Ownership;
  facing: Facing;
}

export interface PredictionResponse {
  predicted_price_rupees: number;
  predicted_price_formatted: string;
  currency: string;
  model_used: string;
  location_recognized: boolean;
}

export const FURNISHING_OPTIONS: Furnishing[] = [
  "Unfurnished",
  "Semi-Furnished",
  "Furnished",
];

export const TRANSACTION_OPTIONS: TransactionType[] = [
  "Resale",
  "New Property",
  "Rent/Lease",
  "Other",
];

export const OWNERSHIP_OPTIONS: Ownership[] = [
  "Freehold",
  "Leasehold",
  "Co-operative Society",
  "Power Of Attorney",
];

export const FACING_OPTIONS: Facing[] = [
  "East",
  "North",
  "North - East",
  "North - West",
  "South",
  "South - East",
  "South -West",
  "West",
];
