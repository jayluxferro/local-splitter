#!/usr/bin/env python3
"""Build a minimal HTML summary from eval JSONL (runs.jsonl).

Usage:
  python scripts/trace_report.py .local_splitter/eval/runs.jsonl -o trace.html
  python scripts/trace_report.py .local_splitter/eval/runs.jsonl   # prints HTML
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize pipeline traces from JSONL.")
    ap.add_argument("jsonl", type=Path, help="Path to runs.jsonl")
    ap.add_argument("-o", "--output", type=Path, help="Write HTML here (default: stdout)")
    args = ap.parse_args()

    stages: Counter[str] = Counter()
    decisions: Counter[tuple[str, str]] = Counter()
    n = 0
    for line in args.jsonl.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        n += 1
        for ev in row.get("trace") or row.get("pipeline_trace") or []:
            st = ev.get("stage", "?")
            stages[st] += 1
            decisions[(st, str(ev.get("decision", "?")))] += 1

    rows = "".join(
        f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in stages.most_common(40)
    )
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>local-splitter trace</title>
<style>body{{font-family:system-ui;margin:2rem}} table{{border-collapse:collapse}}
td,th{{border:1px solid #ccc;padding:0.35rem 0.6rem}}</style></head>
<body>
<h1>local-splitter trace summary</h1>
<p>Lines read: {n} — source: {args.jsonl}</p>
<h2>Stage counts</h2>
<table><thead><tr><th>stage</th><th>count</th></tr></thead><tbody>{rows}</tbody></table>
<h2>Stage × decision (top 30)</h2>
<table><thead><tr><th>stage</th><th>decision</th><th>count</th></tr></thead><tbody>
{"".join(f"<tr><td>{s}</td><td>{d}</td><td>{c}</td></tr>" for (s, d), c in decisions.most_common(30))}
</tbody></table>
</body></html>"""

    if args.output:
        args.output.write_text(html)
    else:
        print(html)


if __name__ == "__main__":
    main()
