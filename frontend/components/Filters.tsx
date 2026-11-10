import type { League } from "@/lib/api";
import { GROUP } from "@/lib/format";

export function Filters({ action, leagues, values, search = true, children }: {
  action: string; leagues: League[]; values: Record<string, string | undefined>; search?: boolean; children?: React.ReactNode;
}) {
  return (
    <form className="filters" action={action}>
      {search && <input type="search" name="q" placeholder="Szukaj zawodnika lub klubu…" defaultValue={values.q ?? ""} aria-label="Szukaj" />}
      <select name="league_id" defaultValue={values.league_id ?? ""} aria-label="Liga">
        <option value="">Wszystkie ligi</option>
        {leagues.map((l) => <option key={l.id} value={l.id}>{l.name}</option>)}
      </select>
      <select name="position_group" defaultValue={values.position_group ?? ""} aria-label="Pozycja">
        <option value="">Wszystkie pozycje</option>
        {Object.entries(GROUP).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
      </select>
      {children}
      <button type="submit">Pokaż</button>
    </form>
  );
}
