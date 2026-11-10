import { API_URL } from "@/lib/api";

// Thin proxy so client components can reach the valuation API without CORS or a public API URL.
async function forward(req: Request, ctx: { params: Promise<{ path: string[] }> }) {
  const { path } = await ctx.params;
  const target = new URL(path.map(encodeURIComponent).join("/"), API_URL.endsWith("/") ? API_URL : API_URL + "/");
  target.search = new URL(req.url).search;
  const res = await fetch(target, {
    method: req.method,
    headers: { "content-type": "application/json" },
    body: req.method === "GET" ? undefined : await req.text(),
    cache: "no-store",
  });
  return new Response(await res.text(), { status: res.status, headers: { "content-type": "application/json" } });
}

export { forward as GET, forward as POST };
