"""Per-episode wall time (from per-episode video mtimes) + concurrency on the same GPU (from queue logs) for a list of
candidates on this node. Usage: pair_timing.py NODE cand1 cand2 ...
Concurrency = number of this node's queue-launched evals whose [EVAL_START, last video mtime] interval overlaps the
25/50/75% points of the candidate's own interval, on the same GPU. Other sessions' evals are not visible here."""
import os, re, sys, glob, datetime as dt, statistics as st
node = sys.argv[1]; cands = sys.argv[2:]
R = '/data/safelab_fr3_eval_20260819/ppu_wave_sample_eval_20260821'
starts = {}
for f in glob.glob(R + '/logs/*queue*.log'):
    for l in open(f, errors='ignore'):
        m = re.match(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\+\d\d:\d\d)? EVAL_START (\S+) gpu=(\d)', l)
        if m:
            t = dt.datetime.fromisoformat(m.group(1))
            if '+08:00' in l: t = t - dt.timedelta(hours=8)   # bjxy logs are UTC+8; normalise to UTC
            starts[m.group(2)] = (t, int(m.group(3)))
def vids(c):
    d = os.path.join(R, 'videos', c)
    if not os.path.isdir(d): return []
    return sorted(os.stat(os.path.join(d, x)).st_mtime for x in os.listdir(d) if x.endswith('.mp4'))
iv = {}
for c, (t0, g) in starts.items():
    v = vids(c)
    if v: iv[c] = (g, t0, dt.datetime.utcfromtimestamp(v[-1]))
for c in cands:
    v = vids(c)
    if len(v) < 5: print(f"{node}\t{c}\tno-videos"); continue
    gaps = [b - a for a, b in zip(v[:-1], v[1:])]
    med = st.median(gaps); p25 = st.quantiles(gaps, n=4)[0]; p75 = st.quantiles(gaps, n=4)[2]
    conc = '?'
    if c in iv:
        g, a, b = iv[c]
        pts = [a + (b - a) * k / 4 for k in (1, 2, 3)]
        conc = '/'.join(str(sum(1 for cc, (gg, aa, bb) in iv.items() if gg == g and aa <= t <= bb)) for t in pts)
    print(f"{node}\t{c}\tepisodes={len(v)}\tmedian={med:.0f}s\tp25={p25:.0f}s\tp75={p75:.0f}s\ttotal={(v[-1]-v[0])/60:.1f}min\tconcurrent_same_gpu={conc}\twindow={dt.datetime.utcfromtimestamp(v[0]).strftime('%m-%d %H:%M')}Z")
