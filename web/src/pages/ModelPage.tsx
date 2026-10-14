import { useMemo, useState } from "react";
import { type Overview, compact, dur, fmtScore } from "../api";
import { BarCI } from "../components/charts";
import { Badges, Status, go, href } from "../components/ui";
import { categoryAverages } from "./Leaderboard";

export default function ModelPage({ ov, mkey }: { ov: Overview; mkey: string }) {
  const m = ov.models.find((x) => x.key === mkey);
  const [tier, setTier] = useState(ov.default_tier);
  const avgs = useMemo(() => categoryAverages(ov, tier).get(mkey), [ov, tier, mkey]);
  const specs = new Map(ov.benchmarks.map((b) => [b.slug, b]));
  const cells = ov.cells.filter((c) => c.model === mkey && c.tier === tier)
    .sort((a, b) => (specs.get(a.benchmark)?.category ?? "").localeCompare(specs.get(b.benchmark)?.category ?? "") || a.benchmark.localeCompare(b.benchmark));
  if (!m) return <div className="empty-state">Unknown model {mkey}</div>;
  const catData = ov.categories.filter((c) => avgs?.has(c)).map((c) => ({ key: c, label: c, value: avgs!.get(c)!.value,
    tip: <><div className="tv">{(avgs!.get(c)!.value * 100).toFixed(1)}</div><div className="tl">{c}</div><div className="tl">{avgs!.get(c)!.n} of {avgs!.get(c)!.of} benchmarks</div></> }));
  return (
    <div className="stack">
      <div>
        <div className="row secondary small"><a href={href()}>Leaderboard</a> / model</div>
        <h1>{m.name}</h1>
        <div className="row small secondary">
          <span className="chip">{m.kind}</span><span className="chip">{m.provider}</span><span className="chip">{compact(m.context)} ctx</span>
          {m.vision && <span className="chip">vision</span>}<code>{m.key}</code>
        </div>
        {m.notes && <p className="small secondary">{m.notes}</p>}
      </div>
      <div className="filters">
        <span className="row" style={{ gap: 0 }}>
          {ov.tiers.map((t) => <button key={t} className={`seg ${t === tier ? "on" : ""}`} onClick={() => setTier(t)}>{t}</button>)}
        </span>
      </div>
      {avgs?.get("__overall") && (
        <div className="card">
          <div className="row" style={{ alignItems: "baseline", gap: 14 }}>
            <span className="hero">{(avgs.get("__overall")!.value * 100).toFixed(1)}</span>
            <span className="secondary">overall · mean of {avgs.get("__overall")!.n} category scores at the {tier} tier</span>
          </div>
        </div>
      )}
      {catData.length > 0 && <div className="card"><h2>Category profile</h2><BarCI data={catData} /></div>}
      <div className="card scroll-x">
        <h2>Benchmarks</h2>
        {cells.length === 0 ? <div className="empty-state">No runs at this tier.</div> : (
          <table className="data">
            <thead><tr><th>benchmark</th><th>category</th><th>status</th><th className="num">score</th><th className="num">95% CI</th><th className="num">n</th><th className="num">errors</th><th className="num">tokens out</th><th className="num">gen time</th><th /></tr></thead>
            <tbody>
              {cells.map((c) => {
                const sp = specs.get(c.benchmark);
                const id = c.id ?? c.live?.run_id;
                return (
                  <tr key={c.benchmark} className="click" onClick={() => id && go(`/run/${id}`)}>
                    <td><a href={href("benchmark", c.benchmark)} onClick={(e) => e.stopPropagation()}>{sp?.title ?? c.benchmark}</a></td>
                    <td className="secondary">{sp?.category}</td><td><Status s={c.status ?? c.live?.status ?? "?"} /></td>
                    <td className="num">{fmtScore(c.score, sp?.score_range)}</td>
                    <td className="num">{c.ci_low != null ? `${fmtScore(c.ci_low, sp?.score_range)}–${fmtScore(c.ci_high, sp?.score_range)}` : ""}</td>
                    <td className="num">{c.n_graded ?? ""}</td><td className="num">{c.n_error || ""}</td>
                    <td className="num">{compact(c.tokens_out)}</td><td className="num">{dur(c.gen_seconds)}</td>
                    <td><Badges badges={sp?.badges} judge={sp?.judge} /></td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
