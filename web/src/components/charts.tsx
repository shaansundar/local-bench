import { useLayoutEffect, useRef, useState, type ReactNode } from "react";
import { useTip } from "./ui";

/* Sequential blue ramp (magnitude). Lightest step = near zero. */
const SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"];
export function seqColor(v: number): { bg: string; fg: string } {
  const i = Math.max(0, Math.min(SEQ.length - 1, Math.round(v * (SEQ.length - 1))));
  return { bg: SEQ[i], fg: i >= 6 ? "#ffffff" : "#0b0b0b" };
}
export function RampLegend({ lo = "0", hi = "100" }: { lo?: string; hi?: string }) {
  return (
    <span className="legend">
      <span>{lo}</span>
      <span className="ramp" aria-hidden>{SEQ.map((c) => <span key={c} style={{ background: c }} />)}</span>
      <span>{hi}</span>
    </span>
  );
}

function useWidth<T extends HTMLElement>(): [React.RefObject<T>, number] {
  const ref = useRef<T>(null);
  const [w, setW] = useState(600);
  useLayoutEffect(() => {
    if (!ref.current) return;
    const ro = new ResizeObserver(([e]) => setW(Math.max(280, e.contentRect.width)));
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);
  return [ref, w];
}

function ticks(max: number, n = 5): number[] {
  const step = max / n;
  const mag = Math.pow(10, Math.floor(Math.log10(step || 1)));
  const nice = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= step) ?? step;
  const out: number[] = [];
  for (let v = 0; v <= max + 1e-9; v += nice) out.push(+v.toFixed(10));
  return out;
}

/* ---- horizontal bars with 95% CI whiskers (single series: models on the category axis) ---- */
export type BarDatum = { key: string; label: string; value: number; lo?: number | null; hi?: number | null; tip?: ReactNode; onClick?: () => void };
export function BarCI({ data, max = 1, fmt = (v: number) => (v * 100).toFixed(1), unit = "" }: { data: BarDatum[]; max?: number; fmt?: (v: number) => string; unit?: string }) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const { show, hide } = useTip();
  const rowH = 30, barH = Math.min(24, rowH - 8), labelW = Math.min(220, width * 0.32), valueW = 64;
  const plotW = Math.max(80, width - labelW - valueW - 8);
  const height = data.length * rowH + 26;
  const x = (v: number) => labelW + (Math.max(0, Math.min(max, v)) / max) * plotW;
  const tk = ticks(max, 4);
  return (
    <div ref={ref}>
      <svg width={width} height={height} role="img" aria-label="bar chart">
        {tk.map((t) => (
          <g key={t}>
            <line className="grid-line" x1={x(t)} x2={x(t)} y1={0} y2={data.length * rowH} />
            <text x={x(t)} y={data.length * rowH + 16} textAnchor="middle">{fmt(t)}{unit}</text>
          </g>
        ))}
        <line className="axis-line" x1={labelW} x2={labelW} y1={0} y2={data.length * rowH} />
        {data.map((d, i) => {
          const y = i * rowH + (rowH - barH) / 2;
          const w = Math.max(0, x(d.value) - labelW);
          const r = Math.min(4, w);
          const path = `M${labelW},${y} h${Math.max(0, w - r)} a${r},${r} 0 0 1 ${r},${r} v${barH - 2 * r} a${r},${r} 0 0 1 ${-r},${r} h${-Math.max(0, w - r)} z`;
          const enter = (e: any) => show(e, d.tip ?? <><div className="tv">{fmt(d.value)}{unit}</div><div className="tl">{d.label}</div></>);
          return (
            <g key={d.key} onClick={d.onClick} style={{ cursor: d.onClick ? "pointer" : undefined }}>
              <text x={labelW - 8} y={i * rowH + rowH / 2 + 4} textAnchor="end">{d.label.length > 30 ? d.label.slice(0, 29) + "…" : d.label}</text>
              <rect x={labelW} y={i * rowH} width={plotW} height={rowH} fill="transparent" tabIndex={0}
                onPointerMove={enter} onPointerLeave={hide} onFocus={enter} onBlur={hide} />
              <path className="bar" d={path} pointerEvents="none" />
              {d.lo != null && d.hi != null && (
                <g pointerEvents="none">
                  <line className="ci" x1={x(d.lo)} x2={x(d.hi)} y1={y + barH / 2} y2={y + barH / 2} />
                  <line className="ci" x1={x(d.lo)} x2={x(d.lo)} y1={y + barH / 2 - 4} y2={y + barH / 2 + 4} />
                  <line className="ci" x1={x(d.hi)} x2={x(d.hi)} y1={y + barH / 2 - 4} y2={y + barH / 2 + 4} />
                </g>
              )}
              <text className="label-strong" x={labelW + plotW + 8} y={i * rowH + rowH / 2 + 4}>{fmt(d.value)}{unit}</text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

/* ---- scatter: one dot per model, direct-labelled (identity by label, single hue) ---- */
export type Pt = { key: string; label: string; x: number; y: number; tip?: ReactNode };
export function Scatter({ points, xLabel, yLabel, yMax = 1, fmtX = (v: number) => String(v), fmtY = (v: number) => (v * 100).toFixed(0) }:
  { points: Pt[]; xLabel: string; yLabel: string; yMax?: number; fmtX?: (v: number) => string; fmtY?: (v: number) => string }) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const { show, hide } = useTip();
  const h = 260, m = { l: 44, r: 16, t: 12, b: 38 };
  const xMax = Math.max(1, ...points.map((p) => p.x)) * 1.08;
  const X = (v: number) => m.l + (v / xMax) * (width - m.l - m.r);
  const Y = (v: number) => m.t + (1 - v / yMax) * (h - m.t - m.b);
  // Direct labels: right of the dot, else left of it, else none (the tooltip still names it) —
  // never clipped by the plot edge, never on top of another label.
  const placed: { x0: number; x1: number; y: number }[] = [];
  const labels = new Map<string, { text: string; x: number; anchor: "start" | "end" }>();
  for (const p of [...points].sort((a, b) => b.y - a.y || a.x - b.x)) {
    const text = p.label.length > 22 ? p.label.slice(0, 21) + "…" : p.label;
    const w = text.length * 6.8, cx = X(p.x), cy = Y(p.y);
    for (const [x0, anchor] of [[cx + 9, "start"], [cx - 9 - w, "end"]] as const) {
      const box = { x0, x1: x0 + w, y: cy };
      const fits = box.x0 >= m.l && box.x1 <= width - m.r;
      const clear = placed.every((o) => Math.abs(o.y - cy) > 14 || box.x1 < o.x0 - 4 || box.x0 > o.x1 + 4);
      const dots = points.every((q) => q === p || Math.abs(Y(q.y) - cy) > 9 || X(q.x) < box.x0 - 6 || X(q.x) > box.x1 + 6);
      if (fits && clear && dots) {
        placed.push(box);
        labels.set(p.key, { text, x: anchor === "start" ? x0 : x0 + w, anchor });
        break;
      }
    }
  }
  return (
    <div ref={ref}>
      <svg width={width} height={h} role="img" aria-label={`${yLabel} vs ${xLabel}`}>
        {ticks(yMax, 4).map((t) => (
          <g key={t}><line className="grid-line" x1={m.l} x2={width - m.r} y1={Y(t)} y2={Y(t)} /><text x={m.l - 6} y={Y(t) + 4} textAnchor="end">{fmtY(t)}</text></g>
        ))}
        {ticks(xMax, 4).map((t) => (
          <text key={t} x={X(t)} y={h - m.b + 16} textAnchor="middle">{fmtX(t)}</text>
        ))}
        <line className="axis-line" x1={m.l} x2={width - m.r} y1={h - m.b} y2={h - m.b} />
        <text x={(width + m.l) / 2} y={h - 4} textAnchor="middle">{xLabel}</text>
        <text x={12} y={(h - m.b) / 2} textAnchor="middle" transform={`rotate(-90 12 ${(h - m.b) / 2})`}>{yLabel}</text>
        {points.map((p) => {
          const enter = (e: any) => show(e, p.tip ?? <><div className="tv">{fmtY(p.y)}</div><div className="tl">{p.label} · {fmtX(p.x)}</div></>);
          return (
            <g key={p.key}>
              <circle className="pt" cx={X(p.x)} cy={Y(p.y)} r={5} pointerEvents="none" />
              {labels.has(p.key) && (
                <text x={labels.get(p.key)!.x} y={Y(p.y) + 4} textAnchor={labels.get(p.key)!.anchor} pointerEvents="none">{labels.get(p.key)!.text}</text>
              )}
              <circle cx={X(p.x)} cy={Y(p.y)} r={12} fill="transparent" tabIndex={0} onPointerMove={enter} onPointerLeave={hide} onFocus={enter} onBlur={hide} />
            </g>
          );
        })}
      </svg>
    </div>
  );
}
