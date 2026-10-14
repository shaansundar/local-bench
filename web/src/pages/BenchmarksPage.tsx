import { type Overview } from "../api";
import { Badges, go, href } from "../components/ui";

export default function BenchmarksPage({ ov }: { ov: Overview }) {
  const runsFor = (slug: string) => new Set(ov.cells.filter((c) => c.benchmark === slug && c.status === "done").map((c) => c.model)).size;
  return (
    <div className="stack">
      <div>
        <h1>Benchmarks</h1>
        <p className="sub">{ov.benchmarks.length} benchmarks in the suite. Add one with <code>bench new &lt;slug&gt;</code> (see docs/NEW_BENCHMARK.md).</p>
      </div>
      {ov.categories.filter((c) => ov.benchmarks.some((b) => b.category === c)).map((cat) => (
        <div className="card scroll-x" key={cat}>
          <h2>{cat}</h2>
          <table className="data" style={{ tableLayout: "fixed" }}>
            <colgroup><col /><col style={{ width: 130 }} /><col style={{ width: 190 }} /><col style={{ width: 200 }} /><col style={{ width: 100 }} /><col style={{ width: 90 }} /></colgroup>
            <thead><tr><th>benchmark</th><th>runner</th><th>metric</th><th>flags</th><th>data</th><th className="num">models run</th></tr></thead>
            <tbody>
              {ov.benchmarks.filter((b) => b.category === cat).sort((a, b) => a.title.localeCompare(b.title)).map((b) => (
                <tr key={b.slug} className="click" onClick={() => go(`/benchmark/${b.slug}`)}>
                  <td><a href={href("benchmark", b.slug)}>{b.title}</a><div className="small muted">{b.description.slice(0, 140)}{b.description.length > 140 ? "…" : ""}</div></td>
                  <td className="small">{b.runner === "chat" ? "direct chat" : b.runner === "oc" ? "opencode agent" : "native harness"}</td>
                  <td className="small">{b.metric}</td>
                  <td><Badges badges={[...new Set([...b.badges, ...(b.requires.vision ? ["vision"] : [])])]} judge={b.judge} /></td>
                  <td className="small">{b.fetched ? <span className="ok">✓ fetched</span> : b.gated ? <span className="secondary">gated</span> : <span className="muted">not fetched</span>}</td>
                  <td className="num">{runsFor(b.slug)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ))}
    </div>
  );
}
