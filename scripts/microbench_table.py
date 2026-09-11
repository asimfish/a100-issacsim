#!/usr/bin/env python3
"""Summarise results/microbench/*.json (+ .meta) from scripts/bench_render.py into markdown tables.

Each JSON holds, for one node/GPU/run: render_only_per_frame, physics_only_per_step_dt1_120, eval_cadence_per_control_step
(median / p10 / p90 ms) and a concurrency note; the .meta file holds the background load sampled during the run.
Usage: python3 scripts/microbench_table.py [results/microbench] > results/microbench/TABLE.md
"""
import glob, json, os, re, sys

d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "results", "microbench")
NODE = {"di-20260831162150-5ll6n": "A100 volc", "ubuntu-TRX50-AI-TOP": "5090 bjxy"}
rows = []
for j in sorted(glob.glob(os.path.join(d, "bench_render_*.json"))):
    r = json.load(open(j)); meta = open(j[:-5] + ".meta").read() if os.path.exists(j[:-5] + ".meta") else ""
    m = re.search(r"bench_render_(.+?)_gpu(\d)_(\d+x\d+)_(\d{4}_\d{6})", os.path.basename(j))
    host, gpu, res, ts = m.groups()
    bg = re.search(r"bg_during_measure: (.*)", meta)
    util = re.search(r"util_mean=(\d+)%", meta); procs = re.search(r"procs_mean=([\d.]+)", meta)
    ub = re.search(r"util_before=(\d+)", meta); pb = re.search(r"procs_on_gpu=(\d+)", meta)
    rows.append(dict(node=NODE.get(host, host), gpu=gpu, res=res, ts=f"{ts[:2]}-{ts[2:4]} {ts[5:7]}:{ts[7:9]}Z",
        util=int(util.group(1)) if util else (int(ub.group(1)) if ub else None),
        procs=float(procs.group(1)) if procs else (int(pb.group(1)) if pb else None), sampled=bool(bg),
        ren=r["render_only_per_frame"], phy=r["physics_only_per_step_dt1_120"], ctl=r["eval_cadence_per_control_step"], n=r["frames"]))

def f(x): return f"{x['median_ms']:.0f} ({x['p10_ms']:.0f}–{x['p90_ms']:.0f})"
print("| 节点 / 卡 | 时间 (UTC) | 同卡背景负载 (util / 进程数) | L0 纯渲染 ms/帧 (3 相机 640×480) | L1 纯物理 ms/步 (dt 1/120) | L2 仿真控制步 ms (4 物理步 + 1 渲染) |")
print("|---|---|---|---|---|---|")
for r in sorted(rows, key=lambda r: (r["node"], r["ts"])):
    load = f"{r['util']}% / {r['procs']}" + ("" if r["sampled"] else "（仅启动前快照）") if r["util"] is not None else "n/a"
    print(f"| {r['node']} GPU{r['gpu']} | {r['ts']} | {load} | {f(r['ren'])} | {f(r['phy'])} | {f(r['ctl'])} |")
print()
print("中位数 (p10–p90)，每项 n=%d 帧/步（物理 4n 步），Kit 预热 30 步后计时，`torch.cuda.synchronize()` 后取 `perf_counter`。" % rows[0]["n"] if rows else "")
print()
# best-of (least contended estimate) per node
print("| 节点 | 各项取全部运行中的最小中位数（最接近独占卡的估计） | L0 渲染 | L1 物理 | L2 控制步 | 由 L2 推算 6 s 集(180 控制步)的纯仿真时间 |")
print("|---|---|---|---|---|---|")
best = {}
for r in rows:
    b = best.setdefault(r["node"], {"ren": 1e9, "phy": 1e9, "ctl": 1e9, "k": 0})
    b["ren"] = min(b["ren"], r["ren"]["median_ms"]); b["phy"] = min(b["phy"], r["phy"]["median_ms"]); b["ctl"] = min(b["ctl"], r["ctl"]["median_ms"]); b["k"] += 1
for node, b in sorted(best.items()):
    print(f"| {node} | {b['k']} 次运行 | {b['ren']:.0f} ms ({1000/b['ren']:.1f} fps) | {b['phy']:.0f} ms ({1000/b['phy']:.0f} 步/s) | {b['ctl']:.0f} ms ({1000/b['ctl']:.1f} 步/s) | {b['ctl']*180/1000:.0f} s |")
if len(best) == 2:
    a, c = best["A100 volc"], best["5090 bjxy"]
    print(f"| **A100 / 5090 倍数** | | **{a['ren']/c['ren']:.2f}×** | **{a['phy']/c['phy']:.2f}×** | **{a['ctl']/c['ctl']:.2f}×** | |")
