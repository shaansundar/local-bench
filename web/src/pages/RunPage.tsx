import { useEffect, useState } from "react";
import { type SampleDetail, type SampleRow, api, compact, dur, fmtScore } from "../api";
import { BarCI } from "../components/charts";
import { Badges, ErrorBox, Loading, Status, href, useAsync } from "../components/ui";

const PAGE = 100;

export default function RunPage({ id }: { id: string }) {
  const run = useAsync(() => api.run(id), [id]);
  const [filter, setFilter] = useState("all");
  const [q, setQ] = useState("");
  const [offset, setOffset] = useState(0);
  const [open, setOpen] = useState<{ sid: string; rep: number } | null>(null);
  const samples = useAsync(() => api.samples(id, { offset: String(offset), limit: String(PAGE), ...(filter !== "all" ? { filter } : {}), ...(q ? { q } : {}) }), [id, filter, q, offset]);
  useEffect(() => {
    if (run.data && ["running", "grading", "queued"].includes(run.data.status)) {
      const t = setInterval(() => { run.reload(); samples.reload(); }, 4000);
      return () => clearInterval(t);
    }
  }, [run.data?.status]); // eslint-disable-line react-hooks/exhaustive-deps

  if (run.error) return <ErrorBox e={run.error} />;
  if (!run.data) return <Loading />;
  const r = run.data;
  const range = r.spec?.score_range ?? [0, 1];
  const metrics = r.metrics ?? {};
  const breakdowns = Object.entries(metrics).filter(([, v]) => v && typeof v === "object" && !Array.isArray(v)
    && Object.values(v as object).every((x) => typeof x === "number")) as [string, Record<string, number>][];
  const scalars = Object.entries(metrics).filter(([k, v]) => typeof v === "number" && k !== "n_scored") as [string, number][];

  return (
    <div className="stack">
      <div>
        <div className="row secondary small">
          <a href={href()}>Leaderboard</a> / <a href={href("benchmark", r.benchmark)}>{r.spec?.title ?? r.benchmark}</a> / <a href={href("model", r.model)}>{r.model}</a>
        </div>
        <h1>{r.spec?.title ?? r.benchmark} · {r.model}</h1>
        <div className="row small secondary">
          <Status s={r.status} /><span className="chip">tier: {r.tier}</span><span className="chip">harness: {r.harness}</span>
          {r.subset_hash && <span className="chip mono">subset {r.subset_hash}</span>}
          {r.judge_id && <span className="chip">judge: {r.judge_id}</span>}
          <Badges badges={r.badges ?? r.spec?.badges} judge={r.spec?.judge} /><span className="mono muted">{r.id}</span>
        </div>
        {r.error && <p className="err small">{r.error}</p>}
      </div>
      <div className="card">
        <div className="kpis">
          <div className="kpi"><div className="hero">{fmtScore(r.score, range)}</div><div className="l">{r.spec?.metric ?? "score"}{r.ci_low != null && ` · 95% CI ${fmtScore(r.ci_low, range)}–${fmtScore(r.ci_high, range)}`}</div></div>
          <div className="kpi"><div className="v">{r.n_graded}/{r.n_total}</div><div className="l">graded</div></div>
          <div className="kpi"><div className={`v ${r.n_error ? "err" : ""}`}>{r.n_error}</div><div className="l">errors</div></div>
          <div className="kpi"><div className="v">{r.n_pending}</div><div className="l">judge pending</div></div>
          <div className="kpi"><div className="v">{compact(r.tokens_out)}</div><div className="l">output tokens</div></div>
          <div className="kpi"><div className="v">{dur(r.gen_seconds)}</div><div className="l">generation time (sum)</div></div>
          {scalars.map(([k, v]) => <div className="kpi" key={k}><div className="v">{v <= 1 && v >= 0 ? (v * 100).toFixed(1) : v.toFixed(2)}</div><div className="l">{k.replace(/_/g, " ")}</div></div>)}
        </div>
      </div>
      {breakdowns.map(([k, v]) => (
        <div className="card" key={k}>
          <h2>{k.replace(/_/g, " ")}</h2>
          <BarCI data={Object.entries(v).sort((a, b) => b[1] - a[1]).map(([lab, val]) => ({ key: lab, label: lab, value: val }))} />
        </div>
      ))}
      <div className="card">
        <div className="filters">
          <span className="row" style={{ gap: 0 }}>
            {["all", "wrong", "right", "error", "pending"].map((f) => <button key={f} className={`seg ${f === filter ? "on" : ""}`} onClick={() => { setFilter(f); setOffset(0); }}>{f}</button>)}
          </span>
          <input type="search" placeholder="search id or output…" value={q} onChange={(e) => { setQ(e.target.value); setOffset(0); }} />
          <span className="spacer" />
          {samples.data && <span className="small secondary">{samples.data.total} samples</span>}
          <button disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - PAGE))}>‹</button>
          <button disabled={!samples.data || offset + PAGE >= samples.data.total} onClick={() => setOffset(offset + PAGE)}>›</button>
        </div>
        <div className="scroll-x">
          <table className="data">
            <thead><tr><th>sample</th><th className="num">rep</th><th>status</th><th className="num">score</th><th>extracted</th><th className="num">out tok</th><th className="num">time</th><th>finish</th></tr></thead>
            <tbody>
              {samples.data?.rows.map((s: SampleRow) => (
                <tr key={`${s.sample_id}|${s.rep}`} className="click" onClick={() => setOpen({ sid: s.sample_id, rep: s.rep })}>
                  <td className="mono small">{s.sample_id}</td><td className="num">{s.rep}</td><td><Status s={s.status} /></td>
                  <td className="num">{s.score == null ? "" : s.score >= 1 ? <span className="ok">✓ 1</span> : s.score <= 0 ? <span className="err">✕ 0</span> : s.score.toFixed(2)}</td>
                  <td className="mono small" style={{ maxWidth: 360, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{s.error ?? (typeof s.extracted === "string" ? s.extracted : s.extracted == null ? "" : JSON.stringify(s.extracted))}</td>
                  <td className="num">{compact(s.usage?.output_tokens)}</td><td className="num">{dur(s.latency)}</td><td className="small secondary">{s.finish_reason}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      {open && <SampleDrawer runId={id} sid={open.sid} rep={open.rep} onClose={() => setOpen(null)} />}
    </div>
  );
}

function Json({ v }: { v: unknown }) {
  return <pre className="block">{typeof v === "string" ? v : JSON.stringify(v, null, 2)}</pre>;
}

function content(c: unknown): string {
  if (typeof c === "string") return c;
  if (Array.isArray(c)) return c.map((p: any) => p.text ?? `[${p.type}]`).join("\n");
  return JSON.stringify(c);
}

function SampleDrawer({ runId, sid, rep, onClose }: { runId: string; sid: string; rep: number; onClose: () => void }) {
  const d = useAsync(() => api.sample(runId, sid, rep), [runId, sid, rep]);
  useEffect(() => {
    const k = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", k);
    return () => window.removeEventListener("keydown", k);
  }, [onClose]);
  return (
    <>
      <div className="drawer-bg" onClick={onClose} />
      <aside className="drawer" role="dialog" aria-label="sample detail">
        <div className="row"><h2 style={{ margin: 0 }}>{sid} <span className="muted">· rep {rep}</span></h2><span className="spacer" /><button onClick={onClose}>Close</button></div>
        {d.error ? <ErrorBox e={d.error} /> : !d.data ? <Loading /> : <SampleBody s={d.data} runId={runId} />}
      </aside>
    </>
  );
}

function SampleBody({ s, runId }: { s: SampleDetail; runId: string }) {
  const turns = (s.extra?.turns as { role: string; content: unknown }[] | undefined) ?? null;
  const key = s.artifacts?.transcript ? s.artifacts.transcript.replace(/^transcripts\//, "").replace(/\.jsonl$/, "") : null;
  return (
    <div className="stack" style={{ marginTop: 12 }}>
      <div className="row"><Status s={s.status} />{s.score != null && <span>score <b>{s.score}</b></span>}<span className="secondary small">{s.finish_reason} · {dur(s.latency)} · {compact(s.usage?.output_tokens)} out / {compact(s.usage?.input_tokens)} in</span></div>
      {s.error && <div><h3>Error</h3><Json v={s.error} /></div>}
      <div className="grid2">
        <div><h3>Extracted</h3><Json v={s.extracted ?? "—"} /></div>
        <div><h3>Gold</h3><Json v={s.gold ?? "—"} /></div>
      </div>
      {s.prompt && (
        <div><h3>Prompt</h3>{s.prompt.map((m, i) => <div className="msg" key={i}><div className="role">{m.role}</div><pre className="block">{content(m.content)}</pre></div>)}</div>
      )}
      {s.reasoning && <details><summary>Reasoning ({compact(s.reasoning.length)} chars)</summary><pre className="block">{s.reasoning}</pre></details>}
      <div><h3>Response</h3><pre className="block">{s.output || "(empty)"}</pre></div>
      {turns && turns.length > 0 && (
        <details><summary>Full conversation ({turns.length} messages)</summary>{turns.map((m, i) => <div className="msg" key={i}><div className="role">{m.role}</div><pre className="block">{content(m.content)}</pre></div>)}</details>
      )}
      {s.judge && <div><h3>Judge</h3><Json v={s.judge} /></div>}
      {s.details && Object.keys(s.details).length > 0 && <details open><summary>Grading details</summary><Json v={s.details} /></details>}
      {key && <Transcript runId={runId} tkey={key} />}
      {s.artifacts?.workspace && <Workspace runId={runId} path={s.artifacts.workspace} />}
    </div>
  );
}

function Transcript({ runId, tkey }: { runId: string; tkey: string }) {
  const t = useAsync(() => api.transcript(runId, tkey), [runId, tkey]);
  return (
    <details open>
      <summary>Agent transcript {t.data && `(${t.data.filter((e) => e.type === "tool").length} tool calls)`}</summary>
      {t.error ? <ErrorBox e={t.error} /> : !t.data ? <Loading /> : t.data.map((e, i) => (
        <div className="ev" key={i}>
          {e.type === "text" && <><div className="h">assistant</div><pre className="block">{e.text}</pre></>}
          {e.type === "tool" && <><div className="h">tool · <b>{e.tool}</b> · {e.status}</div><pre className="block">{JSON.stringify(e.input, null, 2)}</pre>{e.output != null && <pre className="block" style={{ marginTop: 6 }}>{typeof e.output === "string" ? e.output : JSON.stringify(e.output, null, 2)}</pre>}</>}
          {e.type === "step" && <div className="h">step finished: {e.reason} · {e.tokens?.output ?? 0} out tokens</div>}
          {e.type === "error" && <div className="err small">{JSON.stringify(e.error)}</div>}
        </div>
      ))}
    </details>
  );
}

function Workspace({ runId, path }: { runId: string; path: string }) {
  const t = useAsync(() => api.tree(runId, path), [runId, path]);
  return (
    <details>
      <summary>Workspace files</summary>
      {t.data?.length === 0 && <div className="muted small">empty</div>}
      {t.data?.map((f) => <div key={f.path} className="small mono"><a href={api.fileUrl(runId, f.path)} target="_blank" rel="noreferrer">{f.path.replace(path + "/", "")}</a> <span className="muted">{compact(f.size)}B</span></div>)}
    </details>
  );
}
