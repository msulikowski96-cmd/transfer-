import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";
import { api, type Summary } from "@/lib/api";

export const metadata: Metadata = { title: "FMV Intelligence", description: "European football market analytics" };
export const dynamic = "force-dynamic";

export default async function RootLayout({ children }: { children: React.ReactNode }) {
  const summary = await api<Summary>("/summary").catch(() => null);
  return (
    <html lang="pl">
      <body>
        <header className="top">
          <div className="wrap">
            <Link href="/" className="brand">FMV <span>Intelligence</span></Link>
            <nav>
              <Link href="/">Dashboard</Link>
              <Link href="/players">Zawodnicy</Link>
              <Link href="/compare">Porównywarka</Link>
              <Link href="/model">Jakość modelu</Link>
            </nav>
          </div>
        </header>
        {summary?.synthetic_data && (
          <div className="banner">
            Dane syntetyczne: zawodnicy, kluby i kwoty są fikcyjne. Wersja deweloperska do czasu podłączenia licencjonowanych źródeł.
          </div>
        )}
        {summary && !summary.synthetic_data && summary.data_sources.includes("transfermarkt-datasets") && (
          <div className="banner">
            Dane: otwarty zbiór transfermarkt-datasets (do lipca 2026), wyłącznie do celów badawczych. Źródło nie zawiera xG,
            kontraktów ani kontuzji; Ekstraklasa nie ma statystyk zawodników.
          </div>
        )}
        {!summary && <div className="banner">Brak połączenia z API wycen albo brak wytrenowanego modelu.</div>}
        <main className="wrap">{children}</main>
      </body>
    </html>
  );
}
