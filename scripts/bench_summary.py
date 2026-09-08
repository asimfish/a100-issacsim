"""Summarise results/bench/<node>/<mode>: wall time, startup, per-episode / per-epoch time, and background load
(mean GPU util, mean #processes sharing the GPU, 1-min loadavg) sampled every 5 s during the run."""
import os, re, sys, statistics as st
root = sys.argv[1] if len(sys.argv) > 1 else "results/bench"
MODES = ["eval10", "collect10", "rl_cartpole", "rl_franka", "rl_cartpole_cam", "rl_psi_state", "rl_psi_rgb"]
def rd(p): return open(p, encoding="utf-8", errors="replace").read() if os.path.exists(p) else ""
for node in sorted(os.listdir(root)):
    nd = os.path.join(root, node)
    if not os.path.isdir(nd): continue
    print(f"\n### {node}")
    for mode in MODES:
        d = os.path.join(nd, mode)
        if not os.path.isdir(d): continue
        t = rd(os.path.join(d, "timing.txt")); m0 = re.search(r"T0=([\d.]+)", t); m1 = re.search(r"T1=([\d.]+).*rc=(\d+)", t)
        if not (m0 and m1): print(f"{mode}: running/incomplete"); continue
        T0, T1, rc = float(m0.group(1)), float(m1.group(1)), m1.group(2); wall = T1 - T0
        bg = [l.split(",") for l in rd(os.path.join(d, "gpu_samples.csv")).splitlines() if l.count(",") >= 4]
        util = st.mean(float(b[1]) for b in bg) if bg else float("nan"); procs = st.mean(float(b[3]) for b in bg) if bg else float("nan")
        load = st.mean(float(b[4]) for b in bg) if bg else float("nan")
        line = f"{mode}: rc={rc} wall={wall:.0f}s"
        s = rd(os.path.join(d, "summary.log"))
        if mode == "eval10":
            ts = [float(l.split()[0].split("/")[0].rstrip(".")) for l in t.splitlines() if "episode_" in l]
            if len(ts) >= 2:
                gaps = [b - a for a, b in zip(ts, ts[1:])]; per = st.mean(gaps)
                line += f" startup~{ts[0]-T0-per:.0f}s per_episode={per:.1f}s (n={len(gaps)} gaps, range {min(gaps):.0f}-{max(gaps):.0f}) 10ep_total={wall:.0f}s"
        elif mode == "collect10":
            e = [(int(l.split()[0]), int(l.split("=")[1])) for l in rd(os.path.join(nd, "collect10_ep_times.txt")).splitlines()]
            if len(e) >= 2:
                gaps = [(b[0] - a[0]) / (b[1] - a[1]) for a, b in zip(e, e[1:]) if b[1] > a[1]]; per = st.mean(gaps)
                line += f" per_episode={per:.1f}s (n={len(gaps)}) startup~{wall-10*per:.0f}s 10ep_total={wall:.0f}s"
            m = re.findall(r"(\d+/\d+) \u6b21", s) or re.findall(r"\((\d+/\d+)\)", s); line += f" success={m[-1] if m else '?'}"
        else:
            fps = re.findall(r"fps total: (\d+) epoch: (\d+)/(\d+) frames: (\d+)", s)
            if len(fps) >= 3:
                fpe = int(fps[2][3]) - int(fps[1][3]); per = [fpe / int(f[0]) for f in fps[1:]]; tot = sum(fpe / int(f[0]) for f in fps)
                line += f" frames/epoch={fpe} per_epoch={st.mean(per):.2f}s (epochs 2-{len(fps)}) 10ep_train={tot:.1f}s startup+scene~{wall-tot:.0f}s"
            else:
                err = re.findall(r"(OutOfMemory\w*|FileNotFoundError|Traceback)", s); line += f" ERR={err[:1]}"
        line += f" | bg: util={util:.0f}% procs_on_gpu={procs:.1f} load1={load:.0f}"
        print(line)
