import { api, type League, type PlayerRow, type Summary } from "@/lib/api";
import { eur, int, pct } from "@/lib/format";
import { PlayerTable } from "@/components/PlayerTable";
import { Filters } from "@/components/Filters";

type Search = { league_id?: string; position_group?: string; max_age?: string };

export default async function Dashboard({ searchParams }: { searchParams: Promise<Search> }) {
  const sp = await searchParams;
  const [summary, leagues, under, top] = await Promise.all([
    api<Summary>("/summary"), api<League[]>("/leagues"),
    api<PlayerRow[]>("/rankings/undervalued", { ...sp, limit: 15 }),
    api<{ items: PlayerRow[] }>("/players", { limit: 8 }),
  ]);
  const m = summary.test_metrics;
  return (
    <>
      <section>
        <h1>European Football Market Analytics</h1>
        <p className="sub">Wyceny na dzień {summary.as_of} · model {summary.model_version}</p>
      </section>
      <section className="grid g4">
        <div className="card kpi"><div className="label">Piłkarze w bazie</div><div className="value">{int(summary.players)}</div>
          <div className="note">{int(summary.valued_players)} z aktualną wyceną</div></div>
        <div className="card kpi"><div className="label">Ligi / kluby</div><div className="value">{summary.leagues} / {summary.clubs}</div>
          <div className="note">{int(summary.transfers)} transferów w historii</div></div>
        <div className="card kpi"><div className="label">Typowy błąd Transfer Value</div>
          <div className="value accent">{pct(m.transfer_value.median_ape)}</div>
          <div className="note">mediana błędu na {int(m.transfer_value.n)} transferach testowych · MAE {eur(m.transfer_value.mae_eur)}</div></div>
        <div className="card kpi"><div className="label">Pokrycie przedziału 80%</div>
          <div className="value">{pct(m.market_value.interval_coverage)}</div>
          <div className="note">Market Value; wycena referencyjna myli się typowo o {pct(m.reference_valuation.median_ape)}</div></div>
      </section>

      <section className="card">
        <h2>Niedowartościowani zawodnicy</h2>
        <p className="sub" style={{ marginTop: -6 }}>
          Market Value modelu najwyżej ponad zewnętrzną wyceną referencyjną; tylko wyceny z dobrą jakością danych i niewysoką niepewnością,
          gdzie wycena referencyjna leży poniżej dolnej granicy przedziału modelu.
        </p>
        <Filters action="/" leagues={leagues} values={sp} search={false}>
          <select name="max_age" defaultValue={sp.max_age ?? ""} aria-label="Maksymalny wiek">
            <option value="">Każdy wiek</option><option value="21">do 21 lat</option>
            <option value="23">do 23 lat</option><option value="26">do 26 lat</option>
          </select>
        </Filters>
        <div style={{ marginTop: 12 }}><PlayerTable rows={under} ratio /></div>
      </section>

      <section className="card">
        <h2>Najwyżej wyceniani</h2>
        <PlayerTable rows={top.items} />
      </section>
    </>
  );
}
