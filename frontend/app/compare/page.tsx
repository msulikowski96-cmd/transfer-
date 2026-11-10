import Link from "next/link";
import { api, type PlayerProfile, type PlayerRow } from "@/lib/api";
import { eur, num, POSITION, UNCERTAINTY } from "@/lib/format";

export default async function Compare({ searchParams }: { searchParams: Promise<{ ids?: string }> }) {
  const { ids = "" } = await searchParams;
  const wanted = ids.split(",").map((s) => s.trim()).filter((s) => /^\d+$/.test(s)).slice(0, 4);
  let players: PlayerProfile[] = [];
  let error = "";
  if (wanted.length >= 2) {
    try { players = await api<PlayerProfile[]>("/compare", { ids: wanted.join(",") }); }
    catch { error = "Nie znaleziono wyceny dla co najmniej jednego z podanych zawodników."; }
  }
  const top = players.length ? null : await api<{ items: PlayerRow[] }>("/players", { limit: 6 });
  const rows: [string, (p: PlayerProfile) => React.ReactNode][] = [
    ["Wiek", (p) => num(p.age, 1)], ["Pozycja", (p) => POSITION[p.position] ?? p.position],
    ["Klub", (p) => `${p.club ?? "—"} (${p.league ?? "—"})`],
    ["Market Value", (p) => <b>{eur(p.market_value_eur)}</b>],
    ["Przedział 80%", (p) => `${eur(p.market_value_low_eur)} – ${eur(p.market_value_high_eur)}`],
    ["Transfer Value", (p) => eur(p.transfer_value_eur)],
    ["Za 12 mies.", (p) => eur(p.forecast_12m_eur)], ["Za 24 mies.", (p) => eur(p.forecast_24m_eur)],
    ["Wycena referencyjna", (p) => eur(p.reference_value_eur)],
    ["Kontrakt (mies.)", (p) => num(p.details.profile.contract_months, 0)],
    ["Minuty (ost. sezon)", (p) => num(p.details.profile.l1_minutes, 0)],
    ["Gole / 90", (p) => num(p.details.profile.l1_goals90)], ["Asysty / 90", (p) => num(p.details.profile.l1_assists90)],
    ["xG / 90", (p) => num(p.details.profile.l1_xg90)], ["xA / 90", (p) => num(p.details.profile.l1_xa90)],
    ["Indeks gry (skoryg. o ligę)", (p) => num(p.details.profile.l1_idx_adj)],
    ["Dni kontuzji (12 mies.)", (p) => num(p.details.profile.inj_days_365, 0)],
    ["Niepewność", (p) => <span className={`tag ${p.uncertainty}`}>{UNCERTAINTY[p.uncertainty]}</span>],
    ["Najmocniejszy czynnik", (p) => `${p.details.factors[0].factor} (${p.details.factors[0].effect_pct > 0 ? "+" : ""}${p.details.factors[0].effect_pct.toFixed(0)}%)`],
  ];
  return (
    <>
      <section><h1>Porównywarka</h1><p className="sub">Podaj od 2 do 4 identyfikatorów zawodników (numer z adresu profilu).</p></section>
      <section className="card">
        <form className="filters" action="/compare">
          <input type="text" name="ids" defaultValue={ids} placeholder="np. 12,845,3001" aria-label="Identyfikatory zawodników" />
          <button type="submit">Porównaj</button>
        </form>
        {error && <p className="error">{error}</p>}
        {top && !error && (
          <p className="sub">Przykład: <Link href={`/compare?ids=${top.items.slice(0, 3).map((p) => p.id).join(",")}`}>
            {top.items.slice(0, 3).map((p) => p.name).join(" · ")}</Link></p>
        )}
      </section>
      {players.length > 0 && (
        <section className="card scroll">
          <table>
            <thead><tr><th></th>{players.map((p) => <th key={p.id}><Link href={`/players/${p.id}`}>{p.name}</Link></th>)}</tr></thead>
            <tbody>{rows.map(([label, cell]) => (
              <tr key={label}><td className="muted">{label}</td>{players.map((p) => <td key={p.id}>{cell(p)}</td>)}</tr>
            ))}</tbody>
          </table>
        </section>
      )}
    </>
  );
}
