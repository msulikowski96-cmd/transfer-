// Server-side access to the valuation API. The browser never talks to it directly:
// client components go through /api/* (see app/api/[...path]/route.ts).
export const API_URL = process.env.FMV_API_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

export async function api<T>(path: string, params?: Record<string, string | number | undefined>): Promise<T> {
  const url = new URL(path, API_URL);
  for (const [k, v] of Object.entries(params ?? {})) {
    if (v !== undefined && v !== "") url.searchParams.set(k, String(v));
  }
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) throw new ApiError(res.status, `${res.status} ${await res.text()}`);
  return res.json() as Promise<T>;
}

export type PlayerRow = {
  id: number; name: string; age: number; nationality: string; position: string; position_group: string;
  club: string | null; league: string | null; league_id: number | null; club_id: number | null;
  market_value_eur: number; market_value_low_eur: number; market_value_high_eur: number;
  transfer_value_eur: number | null; forecast_12m_eur: number | null; reference_value_eur: number | null;
  data_quality: number; uncertainty: "low" | "medium" | "high"; value_ratio?: number; similarity?: number;
};
export type Factor = { factor: string; effect_pct: number; direction: "positive" | "negative" | "neutral" };
export type Band = { low: number; mid: number; high: number };
export type PlayerProfile = PlayerRow & {
  as_of: string; foot: string; height_cm: number;
  transfer_value_low_eur: number; transfer_value_high_eur: number; forecast_6m_eur: number; forecast_24m_eur: number;
  details: { factors: Factor[]; summary: string; forecast: Record<"6" | "12" | "24", Band>; n_plausible_buyers: number;
             profile: Record<string, number | null> };
  history: { as_of_date: string; market_value_eur: number; market_value_low_eur: number; market_value_high_eur: number;
             reference_value_eur: number | null; in_sample: boolean }[];
  similar: PlayerRow[];
  transfers: { date: string; type: string; fee_eur: number | null; addons_eur: number | null; fee_confirmed: boolean;
               from_club: string | null; to_club: string | null }[];
  seasons: { season: string; club: string; league: string; appearances: number; minutes: number; goals: number;
             assists: number; xg: number; xa: number }[];
};
export type Metrics = { n: number; mae_eur: number; rmse_eur: number; wape: number; median_ape: number;
                        bias_log?: number; interval_coverage?: number; interval_rel_width?: number };
export type Summary = {
  players: number; clubs: number; leagues: number; transfers: number; valued_players: number; as_of: string;
  model_version: string; synthetic_data: boolean;
  test_metrics: { market_value: Metrics; transfer_value: Metrics; reference_valuation: Metrics;
                  split: Record<string, string | number> };
};
export type League = { id: number; name: string; country: string };
export type Club = { id: number; name: string; league_id: number; league: string };
