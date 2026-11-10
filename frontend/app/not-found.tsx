import Link from "next/link";

export default function NotFound() {
  return <section><h1>Nie znaleziono</h1><p className="sub">Ten zawodnik nie ma aktualnej wyceny. <Link href="/players">Wróć do listy →</Link></p></section>;
}
