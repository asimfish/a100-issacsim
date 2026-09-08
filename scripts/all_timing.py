"""All candidates on this node: per-episode median (video mtimes), own-queue concurrency on the same GPU, resolution tag.
Usage: all_timing.py NODE  -> prints TSV sorted by concurrency then date."""
import os, re, sys, glob, datetime as dt, statistics as st
node = sys.argv[1]
R = '/data/safelab_fr3_eval_20260819/ppu_wave_sample_eval_20260821'
starts = {}
for f in glob.glob(R + '/logs/*queue*.log'):
    for l in open(f, errors='ignore'):
        m = re.match(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(\+\d\d:\d\d)? EVAL_START (\S+) gpu=(\d)', l)
        if m:
            t = dt.datetime.fromisoformat(m.group(1))
            if m.group(2) == '+08:00': t -= dt.timedelta(hours=8)
            starts[m.group(3)] = (t, int(m.group(4)))
def vids(c):
    d = os.path.join(R, 'videos', c)
    if not os.path.isdir(d): return []
    return sorted(os.stat(os.path.join(d, x)).st_mtime for x in os.listdir(d) if x.endswith('.mp4'))
iv = {}
for c, (t0, g) in starts.items():
    v = vids(c)
    if len(v) >= 5: iv[c] = (g, t0, dt.datetime.utcfromtimestamp(v[-1]), v)
def img_of(c):
    lg = os.path.join(R, 'logs', c + '.log')
    try:
        s = open(lg, errors='ignore').read(400000)
        return '224' if '224x224' in s else '480'
    except Exception: return '?'
rows = []
for c, (g, a, b, v) in iv.items():
    gaps = [y - x for x, y in zip(v[:-1], v[1:])]
    if len(gaps) < 10: continue
    pts = [a + (b - a) * k / 4 for k in (1, 2, 3)]
    conc = [sum(1 for cc, (gg, aa, bb, _) in iv.items() if gg == g and aa <= t <= bb) for t in pts]
    kind = 'pp' if ('pp' in c.split('-')[2] if len(c.split('-')) > 2 else False) or 'pick' in c else 'grasp'
    rows.append((max(conc), a, c, len(v), st.median(gaps), img_of(c), kind))
rows.sort(key=lambda r: (r[0], r[1]))
print("node\tconc_max\tstart_utc\tcandidate\tepisodes\tmedian_s_per_episode\timg\tkind")
for r in rows:
    print(f"{node}\t{r[0]}\t{r[1].strftime('%m-%d %H:%M')}\t{r[2]}\t{r[3]}\t{r[4]:.0f}\t{r[5]}\t{r[6]}")
