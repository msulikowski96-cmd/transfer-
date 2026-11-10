import Link from "next/link";
import { api, type League, type PlayerRow } from "@/lib/api";
import { int } from "@/lib/format";
import { PlayerTable } from "@/components/PlayerTable";
import { Filters } from "@/components/Filters";

type Search = { q?: string; league_id?: string; position_group?: string; page?: string };
const PAGE = 25;

export default async function Players({ searchParams }: { searchParams: Promise<Search> }) {
  const sp = await searchParams;
  const page = Math.max(1, Number(sp.page) || 1);
  const [leagues, res] = await Promise.all([
    api<League[]>("/leagues"),
    api<{ total: number; items: PlayerRow[] }>("/players", { q: sp.q, league_id: sp.league_id, position_group: sp.position_group,
                                                             limit: PAGE, offset: (page - 1) * PAGE }),
  ]);
  const link = (p: number) => {
    const q = new URLSearchParams(Object.entries({ ...sp, page: String(p) }).filter(([, v]) => v) as [string, string][]);
    return `/players?${q}`;
  };
  const pages = Math.max(1, Math.ceil(res.total / PAGE));
  return (
    <>
      <section><h1>Zawodnicy</h1><p className="sub">{int(res.total)} zawodników z aktualną wyceną</p></section>
      <section className="card">
        <Filters action="/players" leagues={leagues} values={sp} />
        <div style={{ marginTop: 12 }}><PlayerTable rows={res.items} /></div>
        <div className="pager">
          {page > 1 && <Link href={link(page - 1)}>← Poprzednia</Link>}
          <span>Strona {page} z {pages}</span>
          {page < pages && <Link href={link(page + 1)}>Następna →</Link>}
        </div>
      </section>
    </>
  );
}
