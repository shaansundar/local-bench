import { StrictMode, useEffect, useRef } from "react";
import { createRoot } from "react-dom/client";
import { api } from "./api";
import { ErrorBox, Loading, TooltipProvider, href, useAsync, useRoute, useTheme } from "./components/ui";
import BenchmarkPage from "./pages/BenchmarkPage";
import BenchmarksPage from "./pages/BenchmarksPage";
import Leaderboard from "./pages/Leaderboard";
import LivePage, { useLive } from "./pages/LivePage";
import ModelPage from "./pages/ModelPage";
import RunPage from "./pages/RunPage";
import "./styles.css";

function App() {
  const route = useRoute();
  const ov = useAsync(() => api.overview(), []);
  const live = useLive();
  const [theme, setTheme] = useTheme();
  // refresh the overview when a run finishes
  const lastDone = useRef<string | null>(null);
  useEffect(() => {
    const top = live?.recent[0];
    const sig = top ? `${top.id}:${top.status}:${top.finished_at}` : null;
    if (sig && lastDone.current && sig !== lastDone.current) ov.reload();
    lastDone.current = sig;
  }, [live]); // eslint-disable-line react-hooks/exhaustive-deps

  const [page, arg] = route;
  const nav = [["", "Leaderboard"], ["benchmarks", "Benchmarks"], ["live", "Live"]] as const;
  let body;
  if (ov.error) body = <ErrorBox e={ov.error} />;
  else if (!ov.data) body = <Loading />;
  else if (page === "benchmark" && arg) body = <BenchmarkPage ov={ov.data} slug={arg} />;
  else if (page === "benchmarks") body = <BenchmarksPage ov={ov.data} />;
  else if (page === "model" && arg) body = <ModelPage ov={ov.data} mkey={arg} />;
  else if (page === "run" && arg) body = <RunPage id={arg} />;
  else if (page === "live") body = <LivePage snap={live} />;
  else body = <Leaderboard ov={ov.data} />;
  const active = live?.active.filter((r) => r.status === "running").length ?? 0;

  return (
    <div className="shell">
      <header className="topbar">
        <a className="brand" href={href()}>
          <svg width="18" height="18" viewBox="0 0 16 16" aria-hidden><rect x="1" y="9" width="3" height="6" rx="1" fill="var(--accent)" /><rect x="6.5" y="5" width="3" height="10" rx="1" fill="var(--accent)" /><rect x="12" y="1" width="3" height="14" rx="1" fill="var(--accent)" /></svg>
          Local Bench
        </a>
        <nav className="nav">
          {nav.map(([p, label]) => <a key={p} href={`#/${p}`} className={(page ?? "") === p || (p === "" && !page) ? "active" : ""}>{label}</a>)}
        </nav>
        <span className="spacer" />
        <a className="livepill" href="#/live">
          <span className={`dot ${live?.worker_alive ? "on" : ""}`} />
          {live?.worker_alive ? `worker running · ${active} active` : "worker idle"}
        </a>
        <select value={theme} onChange={(e) => setTheme(e.target.value)} aria-label="theme">
          <option value="auto">Auto</option><option value="light">Light</option><option value="dark">Dark</option>
        </select>
      </header>
      {body}
    </div>
  );
}

createRoot(document.getElementById("root")!).render(
  <StrictMode><TooltipProvider><App /></TooltipProvider></StrictMode>,
);
