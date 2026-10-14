import { useMemo, useState } from "react";
import { type Cell, type Overview, type Spec, fmtScore, norm } from "../api";
import { RampLegend, seqColor } from "../components/charts";
import { Badges, go, href, useTip } from "../components/ui";

export type CatAvg = { value: number; n: number; of: number };

export function categoryAverages(ov: Overview, tier: string): Map<string, Map<string, CatAvg>> {
  const specs = new Map(ov.benchmarks.map((b) => [b.slug, b]));
  const out = new Map<string, Map<string, CatAvg>>();
  for (const m of ov.models) {
    const per = new Map<string, number[]>();
    for (const c of ov.cells) {
      if (c.model !== m.key || c.tier !== tier || c.status !== "done" || c.score == null) continue;
      const sp = specs.get(c.benchmark);
      if (!sp) continue;
      const arr = per.get(sp.category) ?? [];
      arr.push(norm(c.score, sp.score_range));
      per.set(sp.category, arr);
    }
    const row = new Map<string, CatAvg>();
    for (const cat of ov.categories) {
      const vals = per.get(cat);
      const of = ov.benchmarks.filter((b) => b.category === cat).length;
      if (vals?.length) row.set(cat, { value: vals.reduce((a, b) => a + b, 0) / vals.length, n: vals.length, of });
    }
    const cats = [...row.values()];
    if (cats.length) row.set("__overall", { value: cats.reduce((a, b) => a + b.value, 0) / cats.length, n: cats.length, of: ov.categories.length });
    out.set(m.key, row);
  }
  return out;
}

export default function Leaderboard({ ov }: { ov: Overview }) {
  const [tier, setTier] = useState(() => {
    const has = (t: string) => ov.cells.some((c) => c.tier === t && c.status === "done");
    return has(ov.default_tier) ? ov.default_tier : ov.tiers.find(has) ?? ov.default_tier;
  });
  const [cat, setCat] = useState("all");
  const [onlyWithResults, setOnly] = useState(true);
  const { show, hide } = useTip();

  const cellMap = useMemo(() => {
    const m = new Map<string, Cell>();
    for (const c of ov.cells) if (c.tier === tier) m.set(`${c.model}|${c.benchmark}`, c);
    return m;
  }, [ov, tier]);
  const avgs = useMemo(() => categoryAverages(ov, tier), [ov, tier]);
  const models = ov.models
    .filter((m) => !onlyWithResults || ov.cells.some((c) => c.model === m.key && c.tier === tier))
    .sort((a, b) => (avgs.get(b.key)?.get("__overall")?.value ?? -1) - (avgs.get(a.key)?.get("__overall")?.value ?? -1));
  const cats = ov.categories.filter((c) => (cat === "all" || c === cat) && ov.benchmarks.some((b) => b.category === c));
  const benchesByCat = cats.map((c) => [c, ov.benchmarks.filter((b) => b.category === c).sort((a, b) => a.title.localeCompare(b.title))] as const);
  const usedCats = ov.categories.filter((c) => [...avgs.values()].some((r) => r.has(c)));

  return (
    <div className="stack">
      <div>
        <h1>Leaderboard</h1>
        <p className="sub">Latest finished run per model × benchmark. Scores are 0–100; a category score is the mean over its benchmarks the model has run at this tier.</p>
      </div>
      <div className="filters">
        <span className="row" style={{ gap: 0 }} role="group" aria-label="tier">
          {ov.tiers.map((t) => <button key={t} className={`seg ${t === tier ? "on" : ""}`} onClick={() => setTier(t)}>{t}</button>)}
        </span>
        <select value={cat} onChange={(e) => setCat(e.target.value)} aria-label="category">
          <option value="all">All categories</option>
          {ov.categories.map((c) => <option key={c} value={c}>{c}</option>)}
        </select>
        <label className="inline"><input type="checkbox" checked={onlyWithResults} onChange={(e) => setOnly(e.target.checked)} /> only models with runs</label>
        <span className="spacer" />
        <RampLegend />
      </div>

      {models.length === 0 ? (
        <div className="card empty-state">
          No runs at the <b>{tier}</b> tier yet. Start one with <code>bench run -m &lt;model&gt; -b &lt;benchmarks&gt; -t {tier}</code>.
        </div>
      ) : (
        <>
          <div className="card scroll-x">
            <h2>By category</h2>
            <table className="matrix">
              <thead>
                <tr>
                  <th />
                  <th className="bench">Overall</th>
                  {usedCats.map((c) => <th key={c} className="bench">{c}</th>)}
                </tr>
              </thead>
              <tbody>
                {models.map((m) => {
                  const row = avgs.get(m.key);
                  const cellFor = (key: string, label: string) => {
                    const a = row?.get(key);
                    if (!a) return <td key={key} className="cell empty">·</td>;
                    const { bg, fg } = seqColor(a.value);
                    const tip = <><div className="tv">{(a.value * 100).toFixed(1)}</div><div className="tl">{m.name} · {label}</div><div className="tl">{a.n} of {a.of} {key === "__overall" ? "categories" : "benchmarks"} run</div></>;
                    return (
                      <td key={key} className={`cell ${key === "__overall" ? "avg" : ""}`} style={{ background: bg, color: fg }} tabIndex={0}
                        onPointerMove={(e) => show(e, tip)} onPointerLeave={hide} onFocus={(e) => show(e, tip)} onBlur={hide}
                        onClick={() => go(`/model/${encodeURIComponent(m.key)}`)}>
                        {(a.value * 100).toFixed(0)}
                      </td>
                    );
                  };
                  return (
                    <tr key={m.key}>
                      <th className="model"><a href={href("model", m.key)}>{m.name}</a></th>
                      {cellFor("__overall", "overall")}
                      {usedCats.map((c) => cellFor(c, c))}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          <div className="card scroll-x">
            <h2>All benchmarks</h2>
            <table className="matrix">
              <thead>
                <tr>
                  <th />
                  {benchesByCat.map(([c, bs]) => <th key={c} colSpan={bs.length} className="group">{c}</th>)}
                </tr>
                <tr>
                  <th />
                  {benchesByCat.flatMap(([, bs]) => bs.map((b) => (
                    <th key={b.slug} className="bench"><a href={href("benchmark", b.slug)} title={b.title}>{b.title}</a></th>
                  )))}
                </tr>
              </thead>
              <tbody>
                {models.map((m) => (
                  <tr key={m.key}>
                    <th className="model"><a href={href("model", m.key)}>{m.name}</a></th>
                    {benchesByCat.flatMap(([, bs]) => bs.map((b) => <MatrixCell key={b.slug} c={cellMap.get(`${m.key}|${b.slug}`)} spec={b} model={m.name} />))}
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="small muted" style={{ marginTop: 10 }}>
              Marks: <b>A</b> adapted harness · <b>S</b> substitute benchmark · <b>J</b> LLM-judge graded · hatched = not applicable to this model · ⟳ in progress.
              {ov.judge_model ? <> Judge: <code>{ov.judge_model}</code>.</> : <> No judge configured — judge-graded samples stay pending.</>}
            </p>
          </div>
        </>
      )}
    </div>
  );
}

function MatrixCell({ c, spec, model }: { c?: Cell; spec: Spec; model: string }) {
  const { show, hide } = useTip();
  if (!c) return <td className="cell empty">·</td>;
  const live = c.live;
  if (c.status !== "done") {
    if (live?.status === "skipped") {
      const tip = <><div className="tv">n/a</div><div className="tl">{model} · {spec.title}</div><div className="tl">{live.error}</div></>;
      return <td className="cell na" tabIndex={0} onPointerMove={(e) => show(e, tip)} onPointerLeave={hide} onFocus={(e) => show(e, tip)} onBlur={hide}>n/a</td>;
    }
    const label = live?.status === "failed" ? "fail" : live ? `⟳` : "·";
    const tip = <><div className="tv">{live?.status ?? "–"}</div><div className="tl">{model} · {spec.title}</div>{live?.error && <div className="tl">{live.error.slice(0, 200)}</div>}{live && <div className="tl">{live.n_graded}/{live.n_total} graded</div>}</>;
    return (
      <td className={`cell empty ${live?.status === "failed" ? "err" : ""}`} tabIndex={0} onPointerMove={(e) => show(e, tip)} onPointerLeave={hide}
        onFocus={(e) => show(e, tip)} onBlur={hide} onClick={() => live && go(`/run/${live.run_id}`)} style={{ cursor: live ? "pointer" : "default" }}>
        {label}
      </td>
    );
  }
  const v = norm(c.score ?? 0, spec.score_range);
  const { bg, fg } = seqColor(v);
  const marks = [(c.badges ?? spec.badges).includes("adapted") ? "A" : "", (c.badges ?? spec.badges).includes("substitute") ? "S" : "", spec.judge ? "J" : ""].join("");
  const tip = (
    <>
      <div className="tv">{fmtScore(c.score, spec.score_range)}{c.ci_low != null && <span style={{ fontWeight: 400 }}> ({fmtScore(c.ci_low, spec.score_range)}–{fmtScore(c.ci_high, spec.score_range)})</span>}</div>
      <div className="tl">{model} · {spec.title}</div>
      <div className="tl">n={c.n_graded}{c.n_error ? ` · ${c.n_error} errors` : ""}{c.n_pending ? ` · ${c.n_pending} judge-pending` : ""} · {c.harness}</div>
      {live && <div className="tl">⟳ newer run {live.status} ({live.n_graded}/{live.n_total})</div>}
    </>
  );
  return (
    <td className="cell" style={{ background: bg, color: fg }} tabIndex={0} onPointerMove={(e) => show(e, tip)} onPointerLeave={hide}
      onFocus={(e) => show(e, tip)} onBlur={hide} onClick={() => go(`/run/${c.id}`)} onKeyDown={(e) => e.key === "Enter" && go(`/run/${c.id}`)}>
      {fmtScore(c.score, spec.score_range).replace(/\.\d$/, "")}
      {(marks || live) && <span className="cell-badge">{marks}{live ? "⟳" : ""}</span>}
    </td>
  );
}

export { Badges };
