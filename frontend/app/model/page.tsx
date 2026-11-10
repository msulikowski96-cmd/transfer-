import { api, type Metrics } from "@/lib/api";
import { eur, int, num, pct } from "@/lib/format";

type Slices = { overall: Metrics; by_position_group: Record<string, Metrics>; by_league: Record<string, Metrics>;
                by_value_bucket: Record<string, Metrics> };
type Model = {
  name: string; created_at: string; train_end: string; val_end: string; params: Record<string, string | number>;
  metrics: {
    split: Record<string, string | number>; market_value: Slices; transfer_value: Slices;
    baselines: Record<string, Metrics>; components: Record<string, Metrics>; without_drift_correction: Record<string, Metrics>;
    ensemble_weights: Record<string, number>; league_difficulty: Record<string, number>; league_movers_used: number;
    potential: Record<string, { n: number; mae_log_model: number; mae_log_no_change_baseline: number; interval_coverage: number }>;
  };
};

function MetricTable({ rows }: { rows: [string, Metrics][] }) {
  return (
    <div className="scroll"><table>
      <thead><tr><th></th><th className="r">n</th><th className="r">MAE</th><th className="r">RMSE</th><th className="r">WAPE</th>
        <th className="r">Median APE</th><th className="r">Pokrycie 80%</th></tr></thead>
      <tbody>{rows.map(([label, m]) => (
        <tr key={label}><td>{label}</td><td className="r">{int(m.n)}</td><td className="r">{eur(m.mae_eur)}</td><td className="r">{eur(m.rmse_eur)}</td>
          <td className="r">{pct(m.wape, 1)}</td><td className="r">{pct(m.median_ape, 1)}</td>
          <td className="r">{m.interval_coverage === undefined ? "—" : pct(m.interval_coverage, 1)}</td></tr>
      ))}</tbody>
    </table></div>
  );
}

export default async function ModelPage() {
  const model = await api<Model>("/model");
  const m = model.metrics;
  const slices = (s: Slices): [string, Metrics][] => [
    ...Object.entries(s.by_position_group), ...Object.entries(s.by_league), ...Object.entries(s.by_value_bucket)];
  return (
    <>
      <section>
        <h1>Jakość modelu</h1>
        <p className="sub">
          Wersja {model.name} · trening do {m.split.train_end} ({int(Number(m.split.n_train))} transferów) · walidacja i kalibracja do {m.split.val_end}
          {" "}({int(Number(m.split.n_val))}) · test {m.split.test_from} – {m.split.test_to} ({int(Number(m.split.n_test))} transferów, których model nie widział)
        </p>
      </section>
      <section className="card">
        <h2>Wynik na zbiorze testowym: przewidywana a rzeczywista kwota transferu</h2>
        <MetricTable rows={[
          ["Transfer Value (znany kupujący i okno)", m.transfer_value.overall],
          ["Market Value (bez wiedzy o transakcji)", m.market_value.overall],
          ["Tylko CatBoost", m.components.catboost], ["Tylko LightGBM", m.components.lightgbm],
          ["Market Value bez korekty dryfu", m.without_drift_correction.market_value],
          ["Punkt odniesienia: wycena referencyjna portalu", m.baselines.reference_valuation],
          ["Punkt odniesienia: mediana dla pozycji", m.baselines.position_median],
        ]} />
        <p className="sub">Wagi ensemble: CatBoost {num(m.ensemble_weights.catboost)} · LightGBM {num(m.ensemble_weights.lightgbm)}.
          Market Value porównujemy z kwotą transferu, bo to jedyna obserwowalna cena; różnica między nimi obejmuje też warunki konkretnej transakcji.</p>
      </section>
      <section className="grid g2">
        <div className="card"><h2>Transfer Value w podziale</h2><MetricTable rows={slices(m.transfer_value)} /></div>
        <div className="card"><h2>Market Value w podziale</h2><MetricTable rows={slices(m.market_value)} /></div>
      </section>
      <section className="grid g2">
        <div className="card">
          <h2>Prognoza zmiany wartości</h2>
          <div className="scroll"><table>
            <thead><tr><th>Horyzont</th><th className="r">n</th><th className="r">Błąd modelu (log)</th><th className="r">Błąd „bez zmian” (log)</th><th className="r">Pokrycie 80%</th></tr></thead>
            <tbody>{Object.entries(m.potential).map(([h, v]) => (
              <tr key={h}><td>{h.replace("m", " mies.")}</td><td className="r">{int(v.n)}</td><td className="r">{num(v.mae_log_model, 3)}</td>
                <td className="r">{num(v.mae_log_no_change_baseline, 3)}</td><td className="r">{pct(v.interval_coverage, 1)}</td></tr>
            ))}</tbody>
          </table></div>
          <p className="sub">Prognoza 6-miesięczna to połowa zmiany 12-miesięcznej (dane sezonowe).</p>
        </div>
        <div className="card">
          <h2>Wyuczona trudność lig</h2>
          <div className="scroll"><table>
            <thead><tr><th>Liga</th><th className="r">Korekta indeksu gry</th></tr></thead>
            <tbody>{Object.entries(m.league_difficulty).sort((a, b) => b[1] - a[1]).map(([l, v]) => (
              <tr key={l}><td>{l}</td><td className={`r ${v >= 0 ? "pos" : "neg"}`}>{v > 0 ? "+" : ""}{num(v)}</td></tr>
            ))}</tbody>
          </table></div>
          <p className="sub">Oszacowana na {int(m.league_movers_used)} zawodnikach, którzy zmienili ligę (odchylenia standardowe indeksu gry).</p>
        </div>
      </section>
    </>
  );
}
