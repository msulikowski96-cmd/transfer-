"use client";
import { useState } from "react";
import type { Club } from "@/lib/api";
import { eur } from "@/lib/format";

type Result = { buyer: { name: string; league: string }; market_value_eur: number; transfer_value_eur: number;
                transfer_value_low_eur: number; transfer_value_high_eur: number; premium_pct: number };

export function Simulator({ playerId, clubs, contractMonths }: { playerId: number; clubs: Club[]; contractMonths: number | null }) {
  const [buyer, setBuyer] = useState(String(clubs[0]?.id ?? ""));
  const [win, setWin] = useState("summer");
  const [contract, setContract] = useState(contractMonths === null ? "" : String(Math.round(contractMonths)));
  const [result, setResult] = useState<Result | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function run(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true); setError("");
    try {
      const res = await fetch("/api/simulate", {
        method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ player_id: playerId, buyer_club_id: Number(buyer), window: win,
                               contract_months: contract === "" ? null : Number(contract) }),
      });
      const body = await res.json();
      if (!res.ok) throw new Error(typeof body.detail === "string" ? body.detail : "Nie udało się policzyć symulacji");
      setResult(body);
    } catch (err) {
      setResult(null); setError(err instanceof Error ? err.message : String(err));
    } finally { setBusy(false); }
  }

  const leagues = [...new Set(clubs.map((c) => c.league))];
  return (
    <div>
      <form className="filters" onSubmit={run}>
        <select value={buyer} onChange={(e) => setBuyer(e.target.value)} aria-label="Klub kupujący">
          {leagues.map((l) => (
            <optgroup key={l} label={l}>
              {clubs.filter((c) => c.league === l).map((c) => <option key={c.id} value={c.id}>{c.name}</option>)}
            </optgroup>
          ))}
        </select>
        <select value={win} onChange={(e) => setWin(e.target.value)} aria-label="Okno transferowe">
          <option value="summer">Okno letnie</option><option value="winter">Okno zimowe</option>
        </select>
        {contractMonths !== null && (
          <input type="number" min={0} max={84} value={contract} onChange={(e) => setContract(e.target.value)}
                 placeholder="Kontrakt (mies.)" aria-label="Pozostały kontrakt w miesiącach" style={{ width: 150 }} />
        )}
        <button type="submit" disabled={busy || !buyer}>{busy ? "Liczę…" : "Symuluj"}</button>
      </form>
      {error && <p className="error">{error}</p>}
      {result && (
        <p style={{ marginBottom: 0 }}>
          {result.buyer.name} ({result.buyer.league}): <b>{eur(result.transfer_value_eur)}</b>{" "}
          <span className="muted">(przedział 80%: {eur(result.transfer_value_low_eur)} – {eur(result.transfer_value_high_eur)})</span>,{" "}
          <span className={result.premium_pct >= 0 ? "pos" : "neg"}>
            {result.premium_pct >= 0 ? "+" : ""}{result.premium_pct.toFixed(0)}%
          </span>{" "}względem Market Value {eur(result.market_value_eur)}.
        </p>
      )}
    </div>
  );
}
