#!/bin/bash
# per-episode wall time from video mtimes: vid_timing.sh VIDEO_DIR
d=$1; python3 - "$d" <<'PY'
import os,sys,statistics as st
d=sys.argv[1]; fs=sorted(f for f in os.listdir(d) if f.endswith('.mp4'))
t=[os.stat(os.path.join(d,f)).st_mtime for f in fs]
g=[b-a for a,b in zip(t[:-1],t[1:])]
import datetime
print(f"{os.path.basename(d)}: episodes={len(fs)} per_episode median={st.median(g):.0f}s p25={st.quantiles(g,n=4)[0]:.0f}s p75={st.quantiles(g,n=4)[2]:.0f}s total={(t[-1]-t[0])/60:.1f}min window={datetime.datetime.utcfromtimestamp(t[0]).strftime('%m-%d %H:%M')}Z-{datetime.datetime.utcfromtimestamp(t[-1]).strftime('%H:%M')}Z")
PY
