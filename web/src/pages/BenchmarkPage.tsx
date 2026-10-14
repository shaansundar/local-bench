import { useMemo, useState } from "react";
import { type Overview, compact, dur, fmtScore, norm } from "../api";
import { BarCI, Scatter } from "../components/charts";
import { Badges, Status, go, href } from "../components/ui";

export default function BenchmarkPage({ ov, slug }: { ov: Overview; slug: string }) {
  const spec = ov.benchmarks.find((b) => b.slug === slug);
  const [tier, setTier] = useState(ov.default_tier);
  const names = new Map(ov.models.map((m) => [m.key, m.name]));
  const done = useMemo(() => ov.cells.filter((c) => c.benchmark === slug && c.tier === tier && c.status === "done" && c.score != null)
    .sort((a, b) => (b.score ?? 0) - (a.score ?? 0)), [ov, slug, tier]);
  const all = ov.cells.filter((c) => c.benchmark === slug);
  if (!spec) return <div className="empty-state">Unknown benchmark {slug}</div>;
  const r = spec.score_range;
  const fmt = (v: number) => fmtScore(v * (r[1] - r[0]) + r[0], r);
  const perSample = (c: typeof done[number], v: number | undefined) => (c.n_done ? (v ?? 0) / c.n_done : 0);

  return (
    <div className="stack">
      <div>
        <div className="row secondary small"><a href={href()}>Leaderboard</a> / {spec.category}</div>
        <h1>{spec.title}</h1>
        <p className="sub" style={{ marginBottom: 6 }}>{spec.description}</p>
        <div className="row small secondary">
          <span className="chip">{spec.runner === "chat" ? "direct chat" : spec.runner === "oc" ? "opencode agent" : "native harness"}</span>
          <span className="chip">metric: {spec.metric}</span>
          {spec.repeats > 1 && <span className="chip">avg@{spec.repeats}</span>}
          {spec.requires.vision && <span className="chip">vision</span>}
          {spec.requires.min_context > 0 && <span className="chip">≥{compact(spec.requires.min_context)} ctx</span>}
          {spec.requires.network && <span className="chip">web access</span>}
          <Badges badges={spec.badges} judge={spec.judge} />
          {!spec.fetched && <span className="chip">data not fetched</span>}
        </div>
        {spec.notes && <p className="small secondary">{spec.notes}</p>}
        <div className="small" style={{ marginTop: 6 }}>
          {spec.sources.map((s) => <div key={s.url}><a href={s.url} target="_blank" rel="noreferrer">{s.url}</a>{s.revision && <span className="muted"> @ {s.revision.slice(0, 10)}</span>}{s.note && <span className="muted"> — {s.note}</span>}</div>)}
        </div>
      </div>
      <div className="filters">
        <span className="row" style={{ gap: 0 }}>
          {ov.tiers.map((t) => <button key={t} className={`seg ${t === tier ? "on" : ""}`} onClick={() => setTier(t)}>{t}</button>)}
        </span>
      </div>
      {done.length === 0 ? <div className="card empty-state">No finished runs at the {tier} tier.</div> : (
        <>
          <div className="card">
            <h2>Score by model <span className="muted small">(95% CI)</span></h2>
            <BarCI fmt={fmt} data={done.map((c) => ({
              key: c.model, label: names.get(c.model) ?? c.model, value: norm(c.score!, r),
              lo: c.ci_low != null ? norm(c.ci_low, r) : null, hi: c.ci_high != null ? norm(c.ci_high, r) : null,
              onClick: () => go(`/run/${c.id}`),
              tip: <><div className="tv">{fmtScore(c.score, r)} <span style={{ fontWeight: 400 }}>({fmtScore(c.ci_low, r)}–{fmtScore(c.ci_high, r)})</span></div><div className="tl">{names.get(c.model) ?? c.model}</div><div className="tl">n={c.n_graded} · {c.harness}</div></>,
            }))} />
          </div>
          <div className="grid2">
            <div className="card">
              <h2>Score vs output tokens per sample</h2>
              <Scatter xLabel="mean output tokens / sample" yLabel="score" fmtX={compact} fmtY={(v) => (v * 100).toFixed(0)}
                points={done.map((c) => ({ key: c.model, label: names.get(c.model) ?? c.model, x: perSample(c, c.tokens_out), y: norm(c.score!, r) }))} />
            </div>
            <div className="card">
              <h2>Score vs generation time per sample</h2>
              <Scatter xLabel="mean seconds / sample" yLabel="score" fmtX={(v) => v.toFixed(0) + "s"} fmtY={(v) => (v * 100).toFixed(0)}
                points={done.map((c) => ({ key: c.model, label: names.get(c.model) ?? c.model, x: perSample(c, c.gen_seconds), y: norm(c.score!, r) }))} />
            </div>
          </div>
        </>
      )}
      <div className="card scroll-x">
        <h2>Runs</h2>
        <table className="data">
          <thead><tr><th>model</th><th>tier</th><th>status</th><th className="num">score</th><th className="num">95% CI</th><th className="num">n</th><th className="num">errors</th><th className="num">tokens out</th><th className="num">gen time</th><th>run</th></tr></thead>
          <tbody>
            {all.map((c) => {
              const id = c.id ?? c.live?.run_id;
              return (
                <tr key={`${c.model}|${c.tier}`} className="click" onClick={() => id && go(`/run/${id}`)}>
                  <td>{names.get(c.model) ?? c.model}</td><td>{c.tier}</td><td><Status s={c.status ?? c.live?.status ?? "?"} /></td>
                  <td className="num">{fmtScore(c.score, r)}</td><td className="num">{c.ci_low != null ? `${fmtScore(c.ci_low, r)}–${fmtScore(c.ci_high, r)}` : ""}</td>
                  <td className="num">{c.n_graded ?? c.live?.n_graded}</td><td className="num">{c.n_error ?? ""}</td>
                  <td className="num">{compact(c.tokens_out)}</td><td className="num">{dur(c.gen_seconds)}</td><td className="mono small">{id}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
