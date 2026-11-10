// Client of the FMV valuation API (backend/fmv/api.py).
// Set EXPO_PUBLIC_API_URL to the deployed API; localhost only works in the web build or an emulator.
export const API_URL = (process.env.EXPO_PUBLIC_API_URL ?? 'http://localhost:8000').replace(/\/$/, '');

export async function api<T>(path: string, params?: Record<string, string | number | undefined>, init?: RequestInit): Promise<T> {
  const url = new URL(API_URL + path);
  for (const [k, v] of Object.entries(params ?? {})) {
    if (v !== undefined && v !== '') url.searchParams.set(k, String(v));
  }
  const res = await fetch(url.toString(), init);
  const body = await res.json().catch(() => null);
  if (!res.ok) {
    const detail = body && typeof body.detail === 'string' ? body.detail : `HTTP ${res.status}`;
    throw new Error(detail);
  }
  return body as T;
}

export type Uncertainty = 'low' | 'medium' | 'high';
export type PlayerRow = {
  id: number; name: string; age: number; position: string; position_group: string;
  club: string | null; league: string | null; club_id: number | null;
  market_value_eur: number; market_value_low_eur: number; market_value_high_eur: number;
  transfer_value_eur: number | null; forecast_12m_eur: number | null; reference_value_eur: number | null;
  data_quality: number; uncertainty: Uncertainty; value_ratio?: number;
};
export type Factor = { factor: string; effect_pct: number; direction: 'positive' | 'negative' | 'neutral' };
export type PlayerProfile = PlayerRow & {
  as_of: string; nationality: string;
  transfer_value_low_eur: number; transfer_value_high_eur: number; forecast_6m_eur: number; forecast_24m_eur: number;
  details: { factors: Factor[]; summary: string; profile: Record<string, number | null> };
  history: { as_of_date: string; market_value_eur: number; reference_value_eur: number | null }[];
  similar: PlayerRow[];
  seasons: { season: string; club: string; league: string; appearances: number; minutes: number; goals: number; assists: number }[];
};
export type Metrics = { n: number; mae_eur: number; wape: number; median_ape: number; interval_coverage?: number };
export type Summary = {
  players: number; valued_players: number; leagues: number; transfers: number; as_of: string; model_version: string;
  synthetic_data: boolean; data_sources: string[];
  test_metrics: { market_value: Metrics; transfer_value: Metrics; reference_valuation: Metrics; split: Record<string, string | number> };
};
export type Club = { id: number; name: string; league: string };
export type Simulation = { buyer: { name: string; league: string }; market_value_eur: number; transfer_value_eur: number;
  transfer_value_low_eur: number; transfer_value_high_eur: number; premium_pct: number };
export type ModelInfo = {
  name: string;
  metrics: {
    split: Record<string, string | number>;
    market_value: { overall: Metrics; by_league: Record<string, Metrics> };
    transfer_value: { overall: Metrics };
    baselines: Record<string, Metrics>;
    league_difficulty: Record<string, number>;
  };
};
