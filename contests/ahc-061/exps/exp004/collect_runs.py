"""exp004 の W&B 結果を再現データとスタンドアロン図表へ書き出す。"""
# ruff: noqa: E501

import csv
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import wandb

ROOT = Path(__file__).resolve().parent
PROJECT = "harui-i-tokyo-metropolitan-university/ahcrl-meta-ahc061"
RUNS = [
    (343, 0.2, "vp8klbbo"),
    (344, 0.229739671, "kxayn2ia"),
    (345, 0.263901582, "fv81dx1t"),
    (346, 0.303143313, "ablick03"),
    (347, 0.348220225, "7arwho3p"),
    (348, 0.4, "a2j921uf"),
]


def svg_chart(*, title: str, x_label: str, log_x: bool, series: list[dict]) -> str:
    """Return a standalone, responsive SVG line chart."""
    points = [point for row in series for point in row["points"]]
    xs = [point["x"] for point in points]
    ys = [point["y"] for point in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    y_pad = (y_max - y_min) * 0.08
    y_min -= y_pad
    y_max += y_pad
    x_values = [x_min, x_max] if log_x else xs
    if log_x:
        x_min, x_max = min(x_values), max(x_values)
        transform = math.log
    else:
        transform = float
    tx_min, tx_max = transform(x_min), transform(x_max)
    width, height = 920, 500
    left, right, top, bottom = 84, 30, 30, 74

    def sx(value: float) -> float:
        return left + (transform(value) - tx_min) / (tx_max - tx_min) * (width - left - right)

    def sy(value: float) -> float:
        return height - bottom - (value - y_min) / (y_max - y_min) * (height - top - bottom)

    y_ticks = [y_min + (y_max - y_min) * index / 4 for index in range(5)]
    x_ticks = (
        [x_min * (x_max / x_min) ** (index / 5) for index in range(6)]
        if log_x
        else [x_min + (x_max - x_min) * index / 5 for index in range(6)]
    )
    grid = "".join(
        f'<line x1="{left}" y1="{sy(value):.1f}" x2="{width - right}" '
        f'y2="{sy(value):.1f}" class="grid"/><text x="{left - 9}" '
        f'y="{sy(value) + 4:.1f}" text-anchor="end">{value / 1000:.0f}k</text>'
        for value in y_ticks
    )
    x_axis = "".join(
        f'<line x1="{sx(value):.1f}" y1="{height - bottom}" x2="{sx(value):.1f}" '
        f'y2="{height - bottom + 5}" class="axis"/><text x="{sx(value):.1f}" '
        f'y="{height - bottom + 23}" text-anchor="middle">'
        f"{value:.3g}</text>"
        for value in x_ticks
    )

    def render_series(row: dict) -> str:
        polyline = " ".join(f"{sx(point['x']):.1f},{sy(point['y']):.1f}" for point in row["points"])
        markers = "".join(
            f'<circle cx="{sx(point["x"]):.1f}" cy="{sy(point["y"]):.1f}" r="4.5" '
            f'fill="{row["color"]}"><title>{row["label"]}: {point["y"]:,.0f}</title></circle>'
            for point in row["points"]
        )
        return f'<polyline points="{polyline}" fill="none" stroke="{row["color"]}" stroke-width="2.5"/>{markers}'

    paths = "".join(render_series(row) for row in series)
    legend = "".join(
        f'<span><i style="background:{row["color"]}"></i>{row["label"]}</span>' for row in series
    )
    return f'''<div class="ahc061-exp004-chart">
  <style>
    .ahc061-exp004-chart {{ color: #111827; font: 14px system-ui, sans-serif; max-width: 920px; }}
    .ahc061-exp004-chart .title {{ font-size: 17px; font-weight: 600; margin: 0 0 8px; }}
    .ahc061-exp004-chart .legend {{ display: flex; flex-wrap: wrap; gap: 6px 16px; margin: 0 0 8px 84px; }}
    .ahc061-exp004-chart .legend i {{ display: inline-block; width: 18px; height: 3px; margin: 0 5px 3px 0; vertical-align: middle; }}
    .ahc061-exp004-chart svg {{ display: block; width: 100%; height: auto; }}
    .ahc061-exp004-chart text {{ fill: #111827; font-size: 12px; }}
    .ahc061-exp004-chart .axis, .ahc061-exp004-chart .grid {{ stroke: #cbd5e1; stroke-width: 1; }}
    .ahc061-exp004-chart .grid {{ opacity: .55; }}
    @media (max-width: 420px) {{ .ahc061-exp004-chart .legend {{ margin-left: 0; }} }}
  </style>
  <div class="title">{title}</div><div class="legend">{legend}</div>
  <svg viewBox="0 0 {width} {height}" role="img" aria-label="{title}">
    {grid}<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" class="axis"/>
    <line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" class="axis"/>
    {x_axis}{paths}
    <text x="{(left + width - right) / 2:.1f}" y="{height - 15}" text-anchor="middle">{x_label}</text>
    <text x="20" y="{height / 2}" text-anchor="middle" transform="rotate(-90 20 {height / 2})">Mean score</text>
  </svg>
</div>'''


def collect(spec: tuple[int, float, str]) -> dict:
    task_id, lr, run_id = spec
    run = wandb.Api().run(f"{PROJECT}/{run_id}")
    history = [
        {
            "env_steps": int(row["summary/cumulative_env_steps"]),
            "mean_score": float(row["episode/score_mean"]),
        }
        for row in run.history(
            samples=2000,
            keys=["summary/cumulative_env_steps", "episode/score_mean"],
            pandas=False,
        )
        if row.get("summary/cumulative_env_steps") is not None
        and row.get("episode/score_mean") is not None
    ]
    summary = run.summary
    return {
        "task_id": task_id,
        "wandb_name": run.name,
        "wandb_run_id": run_id,
        "lr": lr,
        "total_steps_requested": 100000000,
        "total_env_steps": int(summary["summary/cumulative_env_steps"]),
        "final_mean_score": float(summary["episode/score_mean"]),
        "elapsed_sec": float(summary["summary/elapsed_sec"]),
        "history": history,
    }


def main() -> None:
    data_dir = ROOT / "data"
    figures_dir = ROOT / "figures"
    data_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=len(RUNS)) as pool:
        collected = list(pool.map(collect, RUNS))
    (data_dir / "runs.json").write_text(json.dumps(collected, indent=2) + "\n")
    with (data_dir / "final_scores.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[key for key in collected[0] if key != "history"])
        writer.writeheader()
        writer.writerows(
            [{key: value for key, value in row.items() if key != "history"} for row in collected]
        )
    (figures_dir / "lr_vs_final_mean_score.html").write_text(
        svg_chart(
            title="AHC061 Modula + EWMA: learning rate vs. final mean score",
            x_label="Learning rate (log scale)",
            log_x=True,
            series=[
                {
                    "label": "final mean score",
                    "color": "#2563eb",
                    "points": [{"x": row["lr"], "y": row["final_mean_score"]} for row in collected],
                }
            ],
        )
        + "\n"
    )
    (figures_dir / "steps_vs_mean_score.html").write_text(
        svg_chart(
            title="AHC061 Modula + EWMA: learning curves by learning rate",
            x_label="Environment steps",
            log_x=False,
            series=[
                {
                    "label": f"lr={row['lr']:.9g}",
                    "color": color,
                    "points": [
                        {"x": point["env_steps"], "y": point["mean_score"]}
                        for point in row["history"]
                    ],
                }
                for row, color in zip(
                    collected,
                    ["#2563eb", "#d97706", "#059669", "#7c3aed", "#dc2626", "#0891b2"],
                    strict=True,
                )
            ],
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
