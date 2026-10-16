import { useEffect, useState } from "react";
import { type LiveSnapshot, api, dur, fmtScore } from "../api";
import { Meter, Status, go, href } from "../components/ui";

export function useLive(): LiveSnapshot | null {
  const [snap, setSnap] = useState<LiveSnapshot | null>(null);
  useEffect(() => {
    let es: EventSource | null = null;
    let poll: number | undefined;
    try {
      es = new EventSource("/api/live");
      es.onmessage = (e) => setSnap(JSON.parse(e.data));
      es.onerror = () => {
        es?.close();
        es = null;
        poll = window.setInterval(() => fetch("/api/live/snapshot").then((r) => r.json()).then(setSnap).catch(() => {}), 3000);
      };
    } catch { /* EventSource unsupported */ }
    return () => { es?.close(); if (poll) clearInterval(poll); };
  }, []);
  return snap;
}

export default function LivePage({ snap }: { snap: LiveSnapshot | null }) {
  const w = snap?.worker;
  const [log, setLog] = useState<string[]>([]);
  useEffect(() => {
    if (!w?.model) return;
    const load = () => api.serverLog(w.model).then((r) => setLog(r.lines)).catch(() => {});
    load();
    const t = setInterval(load, 5000);
    return () => clearInterval(t);
  }, [w?.model]);
  if (!snap) return <div className="empty-state">Connecting…</div>;
  return (
    <div className="stack">
      <div><h1>Live</h1><p className="sub">Worker state, the queue, and progress, streamed from the results database.</p></div>
      <div className="card">
        {snap.worker_alive && w ? (
          <div className="stack" style={{ gap: 8 }}>
            <div className="row"><span className="dot on" /><b>Worker running</b><span className="secondary">pid {w.pid} · {w.phase}</span></div>
            <div className="kpis">
              {(w.parallel ?? [w]).map((p: Record<string, any>) => (
                <div className="kpi" key={p.model ?? "idle"}>
                  <div className="v">{p.model ?? "–"}</div>
                  <div className="l">{p.benchmark ?? "–"}{p.tier ? ` · ${p.tier}` : ""}{p.phase && p.phase !== "generating" ? ` · ${p.phase}` : ""}</div>
                </div>
              ))}
              <div className="kpi"><div className="v">{(w.tokens_per_s_60s ?? 0).toFixed(1)}</div><div className="l">output tok/s (last 60s{w.parallel ? ", all models" : ""})</div></div>
              {w.started_at && <div className="kpi"><div className="v">{dur(snap.now - w.started_at)}</div><div className="l">worker uptime</div></div>}
            </div>
          </div>
        ) : (
          <div className="row"><span className="dot" /><span className="secondary">No worker running. Start one with <code>bench run …</code> or <code>bench worker --watch</code>.</span></div>
        )}
      </div>
      <div className="card scroll-x">
        <h2>Queue &amp; in progress</h2>
        {snap.active.length === 0 ? <div className="empty-state">Nothing queued.</div> : (
          <table className="data">
            <thead><tr><th>model</th><th>benchmark</th><th>tier</th><th>status</th><th style={{ minWidth: 180 }}>progress</th><th className="num">graded</th><th className="num">errors</th><th className="num">ETA</th></tr></thead>
            <tbody>
              {snap.active.map((r) => {
                const frac = r.n_total ? r.n_graded / r.n_total : 0;
                const perSample = r.n_done ? (snap.now - (r.started_at ?? snap.now)) / r.n_done : 0;
                const eta = r.status === "running" && r.n_done && r.n_total ? perSample * (r.n_total - r.n_done) : null;
                return (
                  <tr key={r.id} className="click" onClick={() => go(`/run/${r.id}`)}>
                    <td>{r.model}</td><td><a href={href("benchmark", r.benchmark)} onClick={(e) => e.stopPropagation()}>{r.benchmark}</a></td><td>{r.tier}</td>
                    <td><Status s={r.status} /></td><td><Meter value={frac} label={`${r.benchmark} progress`} /></td>
                    <td className="num">{r.n_graded}/{r.n_total}</td><td className="num">{r.n_error || ""}</td><td className="num">{eta ? dur(eta) : ""}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>
      <div className="grid2">
        <div className="card">
          <h2>Worker log</h2>
          <pre className="block" style={{ maxHeight: 320 }}>{(w?.recent ?? []).join("\n") || "—"}</pre>
        </div>
        <div className="card">
          <h2>Model server log {w?.model && <span className="muted small">({w.model})</span>}</h2>
          <pre className="block" style={{ maxHeight: 320 }}>{log.slice(-60).join("\n") || "— (external servers log elsewhere)"}</pre>
        </div>
      </div>
      <div className="card scroll-x">
        <h2>Recently finished</h2>
        <table className="data">
          <thead><tr><th>model</th><th>benchmark</th><th>tier</th><th>status</th><th className="num">score</th><th className="num">n</th><th>finished</th><th>note</th></tr></thead>
          <tbody>
            {snap.recent.map((r) => (
              <tr key={r.id} className="click" onClick={() => go(`/run/${r.id}`)}>
                <td>{r.model}</td><td>{r.benchmark}</td><td>{r.tier}</td><td><Status s={r.status} /></td>
                <td className="num">{fmtScore(r.score)}</td><td className="num">{r.n_graded}/{r.n_total}</td>
                <td className="small secondary">{r.finished_at ? dur(snap.now - r.finished_at) + " ago" : ""}</td>
                <td className="small err">{(r.error ?? "").slice(0, 80)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
