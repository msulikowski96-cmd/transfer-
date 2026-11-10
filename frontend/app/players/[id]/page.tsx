import Link from "next/link";
import { notFound } from "next/navigation";
import { api, ApiError, type Club, type PlayerProfile } from "@/lib/api";
import { eur, int, num, POSITION, signedPct, TRANSFER_TYPE, UNCERTAINTY } from "@/lib/format";
import { ValueChart } from "@/components/ValueChart";
import { Simulator } from "@/components/Simulator";
import { PlayerTable } from "@/components/PlayerTable";

export default async function PlayerPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  let p: PlayerProfile;
  try {
    p = await api<PlayerProfile>(`/players/${encodeURIComponent(id)}`);
  } catch (e) {
    if (e instanceof ApiError && (e.status === 404 || e.status === 422)) notFound();
    throw e;
  }
  const clubs = (await api<Club[]>("/clubs")).filter((c) => c.id !== p.club_id);
  const d = p.details;
  const change = (v: number) => {
    const c = (v / p.market_value_eur - 1) * 100;
    return <span className={c > 2 ? "pos" : c < -2 ? "neg" : "muted"}>{signedPct(c)}</span>;
  };
  const maxEffect = Math.max(10, ...d.factors.map((f) => Math.abs(f.effect_pct)));
  return (
    <>
      <section>
        <h1>{p.name}</h1>
        <p className="sub">
          {num(p.age, 1)} lat · {POSITION[p.position] ?? p.position} · {p.club ?? "bez klubu"} ({p.league ?? "—"}) · {p.nationality} ·
          {" "}{p.height_cm} cm · kontrakt: {d.profile.contract_months === null ? "brak danych" : `${num(d.profile.contract_months, 0)} mies.`} ·
          {" "}<Link href={`/compare?ids=${p.id},${p.similar[0]?.id ?? ""}`}>porównaj →</Link>
        </p>
      </section>

      <section className="grid g4">
        <div className="card kpi"><div className="label">Market Value</div><div className="value accent">{eur(p.market_value_eur)}</div>
          <div className="note">przedział 80%: {eur(p.market_value_low_eur)} – {eur(p.market_value_high_eur)}</div></div>
        <div className="card kpi"><div className="label">Transfer Value</div><div className="value">{eur(p.transfer_value_eur)}</div>
          <div className="note">{eur(p.transfer_value_low_eur)} – {eur(p.transfer_value_high_eur)} · typowy kupujący z tej półki cenowej</div></div>
        <div className="card kpi"><div className="label">Prognoza</div>
          <div className="note" style={{ fontSize: 14, color: "var(--text)" }}>
            6 mies.: <b>{eur(p.forecast_6m_eur)}</b> {change(p.forecast_6m_eur)}<br />
            12 mies.: <b>{eur(p.forecast_12m_eur)}</b> {change(p.forecast_12m_eur as number)}<br />
            24 mies.: <b>{eur(p.forecast_24m_eur)}</b> {change(p.forecast_24m_eur)}
          </div></div>
        <div className="card kpi"><div className="label">Pewność wyceny</div>
          <div className="value" style={{ fontSize: 20 }}>
            <span className={`tag ${p.uncertainty}`} style={{ fontSize: 14 }}>niepewność {UNCERTAINTY[p.uncertainty]}</span>
          </div>
          <div className="note">jakość danych {num(p.data_quality * 100, 0)}% · wycena referencyjna {eur(p.reference_value_eur)}</div></div>
      </section>

      <section className="grid g2">
        <div className="card">
          <h2>Dlaczego taka wycena?</h2>
          <p className="summary">{d.summary}</p>
          {d.factors.map((f) => {
            const w = (Math.min(Math.abs(f.effect_pct), maxEffect) / maxEffect) * 50;
            return (
              <div className="factor" key={f.factor}>
                <span>{f.factor}</span>
                <span className="bar"><i style={{ left: f.effect_pct >= 0 ? "50%" : `${50 - w}%`, width: `${w}%`,
                  background: f.direction === "positive" ? "var(--pos)" : f.direction === "negative" ? "var(--neg)" : "var(--muted)" }} /></span>
                <span className={f.direction === "positive" ? "pos" : f.direction === "negative" ? "neg" : "muted"} style={{ textAlign: "right" }}>
                  {signedPct(f.effect_pct)}</span>
              </div>
            );
          })}
          <p className="sub">Wpływ grup cech na wycenę względem przeciętnego transferu (SHAP, komponent CatBoost).</p>
        </div>
        <div className="card">
          <h2>Historia i prognoza wartości</h2>
          <ValueChart player={p} />
        </div>
      </section>

      <section className="card">
        <h2>Symulator transferu</h2>
        <p className="sub" style={{ marginTop: -6 }}>Ile zapłaciłby konkretny klub w konkretnym oknie, przy zadanej długości kontraktu.</p>
        <Simulator playerId={p.id} clubs={clubs} contractMonths={d.profile.contract_months} />
      </section>

      <section className="card"><h2>Podobni zawodnicy</h2><PlayerTable rows={p.similar} /></section>

      <section className="grid g2">
        <div className="card">
          <h2>Sezony</h2>
          <div className="scroll"><table>
            <thead><tr><th>Sezon</th><th>Klub</th><th className="r">Mecze</th><th className="r">Min.</th><th className="r">Gole</th>
              <th className="r">Asysty</th><th className="r">xG</th><th className="r">xA</th></tr></thead>
            <tbody>{p.seasons.map((s, i) => (
              <tr key={i}><td>{s.season}</td><td>{s.club} <span className="muted">· {s.league}</span></td><td className="r">{s.appearances}</td>
                <td className="r">{int(s.minutes)}</td><td className="r">{s.goals}</td><td className="r">{s.assists}</td>
                <td className="r">{num(s.xg, 1)}</td><td className="r">{num(s.xa, 1)}</td></tr>
            ))}</tbody>
          </table></div>
        </div>
        <div className="card">
          <h2>Historia transferów</h2>
          {p.transfers.length === 0 ? <p className="muted">Brak transferów w bazie.</p> : (
            <div className="scroll"><table>
              <thead><tr><th>Data</th><th>Z</th><th>Do</th><th>Rodzaj</th><th className="r">Kwota</th></tr></thead>
              <tbody>{p.transfers.map((t, i) => (
                <tr key={i}><td>{t.date}</td><td>{t.from_club ?? "—"}</td><td>{t.to_club ?? "—"}</td><td>{TRANSFER_TYPE[t.type] ?? t.type}</td>
                  <td className="r">{t.type !== "permanent" ? "—" : t.fee_confirmed ? eur(t.fee_eur) : "nieujawniona"}
                    {t.addons_eur ? <span className="muted"> + {eur(t.addons_eur)} bonusów</span> : null}</td></tr>
              ))}</tbody>
            </table></div>
          )}
        </div>
      </section>
    </>
  );
}
