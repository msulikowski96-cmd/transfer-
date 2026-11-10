import Link from "next/link";
import type { PlayerRow } from "@/lib/api";
import { eur, UNCERTAINTY } from "@/lib/format";

export function PlayerTable({ rows, ratio = false }: { rows: PlayerRow[]; ratio?: boolean }) {
  if (!rows.length) return <p className="muted">Brak zawodników spełniających kryteria.</p>;
  return (
    <div className="scroll">
      <table>
        <thead>
          <tr>
            <th>Zawodnik</th><th>Wiek</th><th>Poz.</th><th>Klub</th><th>Liga</th>
            <th className="r">Market Value</th><th className="r">Transfer Value</th><th className="r">Za 12 mies.</th>
            <th className="r">Wycena ref.</th>{ratio && <th className="r">MV / ref.</th>}<th>Niepewność</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((p) => (
            <tr key={p.id}>
              <td><Link href={`/players/${p.id}`}><b>{p.name}</b></Link></td>
              <td>{p.age.toFixed(1).replace(".", ",")}</td><td>{p.position}</td><td>{p.club ?? "—"}</td><td>{p.league ?? "—"}</td>
              <td className="r"><b>{eur(p.market_value_eur)}</b></td>
              <td className="r">{eur(p.transfer_value_eur)}</td>
              <td className="r">{eur(p.forecast_12m_eur)}</td>
              <td className="r muted">{eur(p.reference_value_eur)}</td>
              {ratio && <td className="r pos">×{(p.value_ratio ?? 0).toFixed(2).replace(".", ",")}</td>}
              <td><span className={`tag ${p.uncertainty}`}>{UNCERTAINTY[p.uncertainty]}</span></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
