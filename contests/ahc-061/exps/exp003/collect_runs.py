"""exp003 の W&B 結果を再現データとスタンドアロン図表へ書き出す。"""
# ruff: noqa: E501

import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import wandb

ROOT = Path(__file__).resolve().parent
PROJECT = "harui-i-tokyo-metropolitan-university/ahcrl-meta-ahc061"
RUNS = [
    (326, 64, 0.3, "4luklyvs"),
    (327, 64, 0.4, "15a5w5fb"),
    (328, 64, 0.5333, "nzl45zty"),
    (329, 64, 0.7111, "1xbn8z2j"),
    (330, 128, 0.3, "n5qxjl7v"),
    (331, 128, 0.4, "aeg6e8op"),
    (332, 128, 0.5333, "3fxwkz5u"),
    (333, 128, 0.7111, "jcbnykzr"),
    (334, 256, 0.3, "cwh743aa"),
    (335, 256, 0.4, "zp6rvg1a"),
    (336, 256, 0.5333, "uqd7d5an"),
    (337, 256, 0.7111, "6wxvl78n"),
]
FLOPS_PER_STEP = {64: 193.0e6, 128: 661.9e6, 256: 2.428e9}


def plot_html(*, title: str, x_label: str, x_mode: str, runs: list[dict]) -> str:
    serialized = json.dumps(runs, separators=(",", ":"))
    return f"""<div class=\"ahc061-exp003-chart\">
  <style>
    .ahc061-exp003-chart {{
      --foreground: #111827; --border: #cbd5e1; --popover: #ffffff; --popover-foreground: #111827;
      --viz-series-1: #2563eb; --viz-series-2: #d97706; --viz-series-3: #059669;
      color: var(--foreground); font: 14px system-ui, sans-serif; max-width: 860px;
    }}
    .ahc061-exp003-chart svg {{ display: block; width: 100%; height: auto; overflow: visible; }}
    .ahc061-exp003-chart .title {{ font-size: 17px; font-weight: 500; margin: 0 0 8px; }}
    .ahc061-exp003-chart .legend {{ display: flex; flex-wrap: wrap; gap: 4px 16px; margin: 0 0 8px 72px; }}
    .ahc061-exp003-chart button {{ appearance: none; border: 0; background: transparent; color: inherit; cursor: pointer; font: inherit; padding: 2px 0; }}
    .ahc061-exp003-chart button[aria-pressed=\"false\"] {{ opacity: .35; text-decoration: line-through; }}
    .ahc061-exp003-chart .swatch {{ display: inline-block; width: 18px; height: 3px; margin: 0 5px 3px 0; vertical-align: middle; }}
    .ahc061-exp003-chart .axis, .ahc061-exp003-chart .grid {{ stroke: var(--border); stroke-width: 1; }}
    .ahc061-exp003-chart .grid {{ opacity: .45; }} .ahc061-exp003-chart text {{ fill: var(--foreground); font-size: 12px; }}
    @media (max-width: 420px) {{ .ahc061-exp003-chart .legend {{ margin-left: 0; }} }}
  </style>
  <div class=\"title\">{title}</div><div class=\"legend\"></div><svg viewBox=\"0 0 860 430\" role=\"img\"></svg>
  <script>
  (() => {{
    const root=document.currentScript.parentElement, svg=root.querySelector('svg'), data={serialized};
    const colors={{64:'var(--viz-series-1)',128:'var(--viz-series-2)',256:'var(--viz-series-3)'}}, dashes=['','7,3','2,3','10,3,2,3'];
    const mode='{x_mode}', margin={{l:76,r:24,t:20,b:62}}, W=860,H=430;
    const series=mode==='lr'
      ? [64,128,256].map(ch=>({{key:`ch${{ch}}`,ch,color:colors[ch],points:data.filter(r=>r.channels===ch)}}))
      : [...new Map(data.map(p=>[`${{p.channels}}-${{p.lr}}`,p])).values()].map(seed=>({{key:`ch${{seed.channels}} · lr=${{seed.lr}}`,ch:seed.channels,lr:seed.lr,color:colors[seed.channels],points:data.filter(p=>p.channels===seed.channels&&p.lr===seed.lr)}}));
    const active=new Set(series.map(s=>s.key)); const ns='http://www.w3.org/2000/svg';
    const add=(tag,attrs,text)=>{{const e=document.createElementNS(ns,tag);Object.entries(attrs).forEach(([k,v])=>e.setAttribute(k,v));if(text)e.textContent=text;svg.append(e);return e;}};
    const legend=root.querySelector('.legend'); series.forEach(s=>{{const b=document.createElement('button');b.setAttribute('aria-pressed','true');b.innerHTML=`<span class=\"swatch\" style=\"background:${{s.color}}\"></span>${{mode==='lr'?'channels '+s.ch:s.key}}`;b.onclick=()=>{{active.has(s.key)?active.delete(s.key):active.add(s.key);b.setAttribute('aria-pressed',active.has(s.key));draw();}};legend.append(b);}});
    const xval=p=>mode==='lr'?p.lr:p.env_steps*({{64:193e6,128:661.9e6,256:2.428e9}}[p.channels]);
    const fmtX=x=>mode==='lr'?x.toPrecision(3):(x/1e15).toFixed(x<1e16?1:0)+'e15';
    function draw() {{
      svg.replaceChildren(); const points=series.flatMap(s=>s.points); const xs=points.map(xval),ys=points.map(p=>p.mean_score);
      const xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);
      const lx=v=>Math.log(v), sx=v=>margin.l+(lx(v)-lx(xmin*.88))/(lx(xmax*1.14)-lx(xmin*.88))*(W-margin.l-margin.r);
      const sy=v=>H-margin.b-(v-(ymin-(ymax-ymin)*.08))/((ymax-ymin)*1.16)*(H-margin.t-margin.b);
      for(let i=0;i<5;i++){{const y=ymin+(ymax-ymin)*i/4, py=sy(y);add('line',{{x1:margin.l,y1:py,x2:W-margin.r,y2:py,class:'grid'}});add('text',{{x:margin.l-8,y:py+4,'text-anchor':'end'}},Math.round(y/1000)+'k');}}
      add('line',{{x1:margin.l,y1:margin.t,x2:margin.l,y2:H-margin.b,class:'axis'}});add('line',{{x1:margin.l,y1:H-margin.b,x2:W-margin.r,y2:H-margin.b,class:'axis'}});
      for(let i=0;i<5;i++){{const x=Math.exp(lx(xmin*.88)+(lx(xmax*1.14)-lx(xmin*.88))*i/4),px=sx(x);add('line',{{x1:px,y1:H-margin.b,x2:px,y2:H-margin.b+5,class:'axis'}});add('text',{{x:px,y:H-margin.b+21,'text-anchor':'middle'}},fmtX(x));}}
      series.forEach(s=>{{if(!active.has(s.key))return;const pts=s.points.map(p=>`${{sx(xval(p))}},${{sy(p.mean_score)}}`).join(' ');add('polyline',{{points:pts,fill:'none',stroke:s.color,'stroke-width':2.5,'stroke-dasharray':s.lr?dashes[[0.3,0.4,0.5333,0.7111].indexOf(s.lr)]:''}});s.points.forEach(p=>{{add('circle',{{cx:sx(xval(p)),cy:sy(p.mean_score),r:mode==='lr'?4.5:1.7,fill:s.color}});}});}});
      add('text',{{x:(margin.l+W-margin.r)/2,y:H-12,'text-anchor':'middle'}},'{x_label}');
      add('text',{{x:18,y:(margin.t+H-margin.b)/2,transform:`rotate(-90 18 ${{(margin.t+H-margin.b)/2}})`,'text-anchor':'middle'}},'Mean score');
    }} draw();
  }})();
  </script>
</div>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=int, nargs="*")
    args = parser.parse_args()
    selected = [spec for spec in RUNS if args.tasks is None or spec[0] in args.tasks]

    def collect(spec: tuple[int, int, float, str]) -> dict:
        task_id, channels, lr, run_id = spec
        api = wandb.Api()
        run = api.run(f"{PROJECT}/{run_id}")
        history = []
        for row in run.scan_history(keys=["summary/cumulative_env_steps", "episode/score_mean"]):
            step, score = row.get("summary/cumulative_env_steps"), row.get("episode/score_mean")
            if step is not None and score is not None:
                history.append({"env_steps": int(step), "mean_score": float(score)})
        if not history:
            raise RuntimeError(f"no score history for {run_id}")
        summary = run.summary
        return {
            "task_id": task_id,
            "wandb_name": run.name,
            "wandb_run_id": run_id,
            "channels": channels,
            "lr": lr,
            "total_steps_requested": 20000000,
            "total_env_steps": int(summary["summary/cumulative_env_steps"]),
            "final_mean_score": float(summary["episode/score_mean"]),
            "elapsed_sec": float(summary["summary/elapsed_sec"]),
            "history": history,
        }

    data_dir = ROOT / "data"
    figures_dir = ROOT / "figures"
    data_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)
    runs_path = data_dir / "runs.json"
    existing = json.loads(runs_path.read_text()) if runs_path.exists() else []
    with ThreadPoolExecutor(max_workers=len(selected)) as pool:
        fresh = list(pool.map(collect, selected))
    by_task = {row["task_id"]: row for row in existing}
    by_task.update({row["task_id"]: row for row in fresh})
    collected = [by_task[task_id] for task_id, *_ in RUNS if task_id in by_task]
    runs_path.write_text(json.dumps(collected, indent=2) + "\n")
    if len(collected) != len(RUNS):
        return
    with (data_dir / "final_scores.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[key for key in collected[0] if key != "history"])
        writer.writeheader()
        writer.writerows(
            [{key: value for key, value in row.items() if key != "history"} for row in collected]
        )
    final_points = [
        {"channels": row["channels"], "lr": row["lr"], "mean_score": row["final_mean_score"]}
        for row in collected
    ]
    curve_points = [
        {
            "channels": row["channels"],
            "lr": row["lr"],
            "mean_score": point["mean_score"],
            "env_steps": point["env_steps"],
        }
        for row in collected
        for point in row["history"]
    ]
    (figures_dir / "lr_vs_final_mean_score.html").write_text(
        plot_html(
            title="AHC061 Modula: learning rate vs. final mean_score",
            x_label="Learning rate (log scale)",
            x_mode="lr",
            runs=final_points,
        )
        + "\n"
    )
    (figures_dir / "flops_vs_mean_score.html").write_text(
        plot_html(
            title="AHC061 Modula: learning curves by training FLOPs",
            x_label="Cumulative training FLOPs (log scale; optimizer/environment excluded)",
            x_mode="flops",
            runs=curve_points,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
