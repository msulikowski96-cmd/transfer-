const nf1 = new Intl.NumberFormat("pl-PL", { minimumFractionDigits: 1, maximumFractionDigits: 1 });
const nf0 = new Intl.NumberFormat("pl-PL", { maximumFractionDigits: 0 });

export function eur(v: number | null | undefined): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  if (Math.abs(v) >= 1e6) return `€${nf1.format(v / 1e6)} mln`;
  return `€${nf0.format(v / 1e3)} tys.`;
}
export const pct = (v: number | null | undefined, digits = 0) =>
  v === null || v === undefined ? "—" : `${(v * 100).toFixed(digits).replace(".", ",")}%`;
export const signedPct = (v: number) => `${v > 0 ? "+" : ""}${v.toFixed(0)}%`;
export const num = (v: number | null | undefined, digits = 2) =>
  v === null || v === undefined ? "—" : v.toFixed(digits).replace(".", ",");
export const int = (v: number) => nf0.format(v);

export const POSITION: Record<string, string> = {
  GK: "Bramkarz", CB: "Środkowy obrońca", FB: "Boczny obrońca", DM: "Defensywny pomocnik",
  CM: "Środkowy pomocnik", AM: "Ofensywny pomocnik", W: "Skrzydłowy", ST: "Napastnik",
};
export const GROUP: Record<string, string> = { GK: "Bramkarze", DEF: "Obrońcy", MID: "Pomocnicy", FWD: "Napastnicy" };
export const UNCERTAINTY: Record<string, string> = { low: "niska", medium: "średnia", high: "wysoka" };
export const TRANSFER_TYPE: Record<string, string> = {
  permanent: "Transfer definitywny", loan: "Wypożyczenie", loan_return: "Powrót z wypożyczenia", free: "Wolny transfer",
};
