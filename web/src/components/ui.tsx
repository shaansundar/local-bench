import { createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode } from "react";

/* ---- hash router ------------------------------------------------------------------------- */
export function useRoute(): string[] {
  const [hash, setHash] = useState(() => window.location.hash);
  useEffect(() => {
    const on = () => setHash(window.location.hash);
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return hash.replace(/^#\/?/, "").split("/").filter(Boolean).map(decodeURIComponent);
}
export const go = (path: string) => { window.location.hash = path; };
export const href = (...parts: string[]) => "#/" + parts.map(encodeURIComponent).join("/");

/* ---- tooltip (one shared instance; text via React, never innerHTML) ---------------------- */
type Tip = { x: number; y: number; content: ReactNode } | null;
const TipCtx = createContext<(t: Tip) => void>(() => {});
export function TooltipProvider({ children }: { children: ReactNode }) {
  const [tip, setTip] = useState<Tip>(null);
  const ref = useRef<HTMLDivElement>(null);
  const [pos, setPos] = useState({ left: 0, top: 0 });
  useEffect(() => {
    if (!tip || !ref.current) return;
    const r = ref.current.getBoundingClientRect();
    let left = tip.x + 14, top = tip.y + 14;
    if (left + r.width > window.innerWidth - 8) left = tip.x - r.width - 14;
    if (top + r.height > window.innerHeight - 8) top = tip.y - r.height - 14;
    setPos({ left: Math.max(8, left), top: Math.max(8, top) });
  }, [tip]);
  return (
    <TipCtx.Provider value={setTip}>
      {children}
      {tip && <div className="tooltip" ref={ref} style={pos} role="tooltip">{tip.content}</div>}
    </TipCtx.Provider>
  );
}
export function useTip() {
  const set = useContext(TipCtx);
  const show = useCallback((e: { clientX: number; clientY: number } | React.FocusEvent, content: ReactNode) => {
    if ("clientX" in e) set({ x: e.clientX, y: e.clientY, content });
    else {
      const r = (e.target as Element).getBoundingClientRect();
      set({ x: r.right, y: r.top, content });
    }
  }, [set]);
  const hide = useCallback(() => set(null), [set]);
  return { show, hide };
}

/* ---- small pieces ------------------------------------------------------------------------- */
const STATUS: Record<string, { color: string; label: string; icon: string }> = {
  done: { color: "var(--good)", label: "done", icon: "✓" },
  running: { color: "var(--accent)", label: "running", icon: "▶" },
  grading: { color: "var(--accent)", label: "grading", icon: "…" },
  queued: { color: "var(--baseline)", label: "queued", icon: "◷" },
  paused: { color: "var(--warning)", label: "paused", icon: "❚❚" },
  failed: { color: "var(--critical)", label: "failed", icon: "✕" },
  skipped: { color: "var(--baseline)", label: "skipped", icon: "–" },
  error: { color: "var(--critical)", label: "error", icon: "✕" },
  graded: { color: "var(--good)", label: "graded", icon: "✓" },
  judge_pending: { color: "var(--warning)", label: "judge pending", icon: "◷" },
  generated: { color: "var(--accent)", label: "generated", icon: "…" },
};
export function Status({ s }: { s: string }) {
  const st = STATUS[s] ?? { color: "var(--baseline)", label: s, icon: "•" };
  return (
    <span className="status" title={st.label}>
      <span className="ic" style={{ background: st.color }} aria-hidden />
      <span aria-hidden className="small">{st.icon}</span> {st.label}
    </span>
  );
}

const BADGE_TEXT: Record<string, string> = {
  adapted: "adapted harness", substitute: "substitute", judge: "LLM judge", vision: "vision", partial: "partial",
};
export function Badges({ badges, judge }: { badges?: string[] | null; judge?: boolean }) {
  const all = [...(badges ?? [])];
  if (judge && !all.includes("judge")) all.push("judge");
  if (!all.length) return null;
  return <span className="row" style={{ gap: 4 }}>{all.map((b) => <span className="chip" key={b}>{BADGE_TEXT[b] ?? b}</span>)}</span>;
}

export function Meter({ value, label }: { value: number; label?: string }) {
  const pct = Math.max(0, Math.min(1, value)) * 100;
  return (
    <div className="meter" role="progressbar" aria-valuenow={Math.round(pct)} aria-valuemin={0} aria-valuemax={100} aria-label={label}>
      <div style={{ width: `${pct}%` }} />
    </div>
  );
}

export function Loading() { return <div className="empty-state">Loading…</div>; }
export function ErrorBox({ e }: { e: unknown }) { return <div className="empty-state err">{String(e)}</div>; }

export function useAsync<T>(fn: () => Promise<T>, deps: unknown[]): { data: T | null; error: unknown; reload: () => void } {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [n, setN] = useState(0);
  useEffect(() => {
    let alive = true;
    fn().then((d) => { if (alive) { setData(d); setError(null); } }).catch((e) => alive && setError(e));
    return () => { alive = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, n]);
  return { data, error, reload: () => setN((x) => x + 1) };
}

export function useTheme(): [string, (t: string) => void] {
  const [t, setT] = useState<string>(() => { try { return localStorage.getItem("theme") || "auto"; } catch { return "auto"; } });
  useEffect(() => {
    const root = document.documentElement;
    if (t === "auto") root.removeAttribute("data-theme"); else root.setAttribute("data-theme", t);
    try { localStorage.setItem("theme", t); } catch { /* storage unavailable */ }
  }, [t]);
  return [t, setT];
}
