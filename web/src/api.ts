export type Spec = {
  slug: string; title: string; category: string; description: string; runner: "chat" | "oc" | "native";
  metric: string; judge: boolean; repeats: number; badges: string[]; score_range: [number, number];
  requires: { vision: boolean; min_context: number; tool_calling: boolean; network: boolean };
  sources: { url: string; revision?: string | null; note?: string | null }[]; gated: boolean; fetched: boolean;
  notes?: string | null; license?: string | null;
};
export type ModelInfo = { key: string; name: string; kind: string; provider: string; context: number; vision: boolean; notes?: string | null; has_results: boolean };
export type Run = {
  id: string; model: string; benchmark: string; tier: string; status: string; score: number | null;
  ci_low: number | null; ci_high: number | null; n_total: number; n_done: number; n_graded: number; n_error: number;
  n_pending: number; metrics: Record<string, unknown> | null; badges: string[] | null; harness: string | null;
  judge_id: string | null; subset_hash: string | null; tokens_in: number; tokens_out: number; gen_seconds: number;
  created_at: number; started_at: number | null; finished_at: number | null; updated_at: number | null; error: string | null;
};
export type Cell = Partial<Run> & { model: string; benchmark: string; tier: string; live?: { run_id: string; status: string; n_graded: number; n_total: number; error: string | null } | null };
export type Overview = { categories: string[]; tiers: string[]; default_tier: string; judge_model: string | null; models: ModelInfo[]; benchmarks: Spec[]; cells: Cell[] };
export type SampleRow = { sample_id: string; rep: number; status: string; score: number | null; extracted: unknown; latency: number | null; usage: Record<string, number> | null; finish_reason: string | null; error: string | null; artifacts: Record<string, string> | null };
export type SampleDetail = SampleRow & { prompt: { role: string; content: unknown }[] | null; output: string | null; reasoning: string | null; gold: unknown; details: Record<string, unknown> | null; judge: Record<string, unknown> | null; extra: Record<string, unknown> | null };
export type LiveSnapshot = { now: number; worker_alive: boolean; worker: Record<string, any> | null; active: Run[]; recent: Run[] };

async function get<T>(path: string): Promise<T> {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json() as Promise<T>;
}

export const api = {
  overview: () => get<Overview>("/api/overview"),
  runs: (q: Record<string, string> = {}) => get<Run[]>(`/api/runs?${new URLSearchParams(q)}`),
  run: (id: string) => get<Run & { spec: Spec | null }>(`/api/runs/${encodeURIComponent(id)}`),
  samples: (id: string, q: Record<string, string>) =>
    get<{ total: number; rows: SampleRow[] }>(`/api/runs/${encodeURIComponent(id)}/samples?${new URLSearchParams(q)}`),
  sample: (id: string, sid: string, rep: number) =>
    get<SampleDetail>(`/api/runs/${encodeURIComponent(id)}/samples/${encodeURIComponent(sid)}?rep=${rep}`),
  transcript: (id: string, key: string) => get<any[]>(`/api/runs/${encodeURIComponent(id)}/transcript/${encodeURIComponent(key)}`),
  tree: (id: string, path: string) => get<{ path: string; size: number }[]>(`/api/runs/${encodeURIComponent(id)}/tree/${path}`),
  fileUrl: (id: string, path: string) => `/api/runs/${encodeURIComponent(id)}/files/${path.split("/").map(encodeURIComponent).join("/")}`,
  serverLog: (key: string) => get<{ lines: string[] }>(`/api/servers/${encodeURIComponent(key)}/log`),
};

export function fmtScore(v: number | null | undefined, range: [number, number] = [0, 1]): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "–";
  if (range[1] === 1 && range[0] === 0) return (v * 100).toFixed(1);
  return v.toFixed(2);
}
export function norm(v: number, range: [number, number] = [0, 1]): number {
  return Math.max(0, Math.min(1, (v - range[0]) / (range[1] - range[0] || 1)));
}
export function compact(n: number | null | undefined): string {
  if (n === null || n === undefined) return "–";
  const a = Math.abs(n);
  if (a >= 1e9) return (n / 1e9).toFixed(1) + "B";
  if (a >= 1e6) return (n / 1e6).toFixed(1) + "M";
  if (a >= 1e3) return (n / 1e3).toFixed(1) + "K";
  return String(Math.round(n));
}
export function dur(s: number | null | undefined): string {
  if (!s && s !== 0) return "–";
  if (s < 60) return `${s.toFixed(0)}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`;
  return `${Math.floor(s / 3600)}h ${Math.round((s % 3600) / 60)}m`;
}
export function ago(ts: number | null | undefined): string {
  if (!ts) return "–";
  return dur(Date.now() / 1000 - ts) + " ago";
}
