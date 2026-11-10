import type { PlayerProfile } from "@/lib/api";
import { eur } from "@/lib/format";

// History of the model's Market Value (with the 80% band), the external reference
// valuation, and the 6/12/24-month forecast. Log scale: values span orders of magnitude.
export function ValueChart({ player }: { player: PlayerProfile }) {
  const W = 760, H = 280, L = 70, R = 16, T = 12, B = 28;
  const t = (d: string) => new Date(d).getTime();
  const month = 30.44 * 864e5;
  const now = t(player.as_of);
  const hist = player.history.map((h) => ({ ...h, x: t(h.as_of_date) }));
  const fc = (["6", "12", "24"] as const).map((m) => ({ x: now + Number(m) * month, ...player.details.forecast[m] }));
  const values = [...hist.flatMap((h) => [h.market_value_low_eur, h.market_value_high_eur, h.reference_value_eur ?? h.market_value_eur]),
                  ...fc.flatMap((f) => [f.low, f.high])].filter((v) => v > 0);
  const lo = Math.log(Math.min(...values) * 0.85), hi = Math.log(Math.max(...values) * 1.15);
  const x0 = Math.min(...hist.map((h) => h.x), now - 12 * month), x1 = now + 24 * month;
  const sx = (x: number) => L + ((x - x0) / (x1 - x0)) * (W - L - R);
  const sy = (v: number) => T + (1 - (Math.log(v) - lo) / (hi - lo)) * (H - T - B);
  const line = (pts: { x: number; v: number }[]) => pts.map((p, i) => `${i ? "L" : "M"}${sx(p.x).toFixed(1)},${sy(p.v).toFixed(1)}`).join(" ");
  const area = (pts: { x: number; low: number; high: number }[]) =>
    pts.map((p, i) => `${i ? "L" : "M"}${sx(p.x).toFixed(1)},${sy(p.high).toFixed(1)}`).join(" ") + " " +
    [...pts].reverse().map((p) => `L${sx(p.x).toFixed(1)},${sy(p.low).toFixed(1)}`).join(" ") + " Z";
  const last = hist[hist.length - 1];
  const fcBand = [{ x: now, low: last.market_value_eur, high: last.market_value_eur }, ...fc];
  const ref = hist.filter((h) => h.reference_value_eur).map((h) => ({ x: h.x, v: h.reference_value_eur as number }));
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((k) => Math.exp(lo + k * (hi - lo)));
  const years: number[] = [];
  for (let y = new Date(x0).getFullYear() + 1; y <= new Date(x1).getFullYear(); y++) years.push(y);
  return (
    <>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label="Historia i prognoza wartości rynkowej">
        {ticks.map((v) => (
          <g key={v}><line x1={L} x2={W - R} y1={sy(v)} y2={sy(v)} stroke="#263242" />
            <text x={L - 8} y={sy(v) + 4} textAnchor="end">{eur(v)}</text></g>
        ))}
        {years.map((y) => <text key={y} x={sx(new Date(`${y}-01-01`).getTime())} y={H - 8} textAnchor="middle">{y}</text>)}
        <line x1={sx(now)} x2={sx(now)} y1={T} y2={H - B} stroke="#93a1b3" strokeDasharray="3 4" />
        <path d={area(hist.map((h) => ({ x: h.x, low: h.market_value_low_eur, high: h.market_value_high_eur })))} fill="#3ddc97" opacity=".13" />
        <path d={area(fcBand)} fill="#5aa9ff" opacity=".16" />
        {ref.length > 1 && <path d={line(ref)} fill="none" stroke="#93a1b3" strokeWidth="1.5" strokeDasharray="5 4" />}
        <path d={line(hist.map((h) => ({ x: h.x, v: h.market_value_eur })))} fill="none" stroke="#3ddc97" strokeWidth="2.2" />
        <path d={line([{ x: now, v: last.market_value_eur }, ...fc.map((f) => ({ x: f.x, v: f.mid }))])} fill="none" stroke="#5aa9ff" strokeWidth="2.2" strokeDasharray="6 4" />
        {hist.map((h) => (
          <circle key={h.as_of_date} cx={sx(h.x)} cy={sy(h.market_value_eur)} r="3.5" fill={h.in_sample ? "#131a23" : "#3ddc97"} stroke="#3ddc97" strokeWidth="1.5">
            <title>{`${h.as_of_date}: ${eur(h.market_value_eur)} (${eur(h.market_value_low_eur)} – ${eur(h.market_value_high_eur)})`}</title>
          </circle>
        ))}
        {fc.map((f) => (
          <circle key={f.x} cx={sx(f.x)} cy={sy(f.mid)} r="3.5" fill="#5aa9ff"><title>{`${eur(f.mid)} (${eur(f.low)} – ${eur(f.high)})`}</title></circle>
        ))}
      </svg>
      <div className="legend">
        <span><i style={{ background: "#3ddc97" }} />Market Value (pas: przedział 80%)</span>
        <span><i style={{ background: "#5aa9ff" }} />Prognoza 6 / 12 / 24 mies.</span>
        <span><i style={{ background: "#93a1b3" }} />Wycena referencyjna</span>
        <span>○ wycena z okresu treningowego modelu</span>
      </div>
    </>
  );
}
