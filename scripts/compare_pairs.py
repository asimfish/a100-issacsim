"""Quality comparison of the same eval episode rendered on volc (A100, no RT cores) and bjxy (RTX 5090).

For each pairs/<tag>_ep<NNN>_{volc,bjxy}.mp4:
  * SSIM / PSNR over the first 10 frames (identical initial layout: same eval seed => same object offsets)
  * SSIM / PSNR over the whole clip (informational only; the two policies' actions diverge slightly)
  * side_by_side/<tag>_ep<NNN>.mp4 : labelled left/right video (labels drawn with PIL, then encoded with ffmpeg)
Writes quality_table.md.
"""
import glob, os, re, subprocess, tempfile
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.expanduser("~/Desktop/research/volc_render_check")
os.chdir(ROOT); os.makedirs("side_by_side", exist_ok=True)

def metric(a, b, filt, nframes=None):
    if nframes:
        lavfi = f"[0:v]trim=end_frame={nframes},setpts=PTS-STARTPTS[a];[1:v]trim=end_frame={nframes},setpts=PTS-STARTPTS[b];[a][b]{filt}"
    else:
        lavfi = f"[0:v][1:v]{filt}"
    out = subprocess.run(["ffmpeg", "-v", "info", "-i", a, "-i", b, "-lavfi", lavfi, "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    if filt == "ssim":
        m = re.search(r"All:([\d.]+)", out); return float(m.group(1)) if m else None
    m = re.search(r"average:([\d.]+)", out); return float(m.group(1)) if m else None

def label_bar(text, width=640, height=28):
    img = Image.new("RGB", (width, height), (20, 20, 20)); d = ImageDraw.Draw(img)
    try: font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
    except Exception: font = ImageFont.load_default()
    d.text((10, 5), text, fill=(255, 230, 80), font=font); return img

def side_by_side(a, b, out):
    if os.path.exists(out): return
    tmp = tempfile.mkdtemp()
    bar = Image.new("RGB", (1280, 28)); bar.paste(label_bar("volc  A100-80G  (no RT cores, Vulkan/CUDA RT)"), (0, 0)); bar.paste(label_bar("bjxy  RTX 5090  (RT cores)"), (640, 0))
    bar_p = os.path.join(tmp, "bar.png"); bar.save(bar_p)
    dur = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", a], capture_output=True, text=True).stdout.strip()
    dur = f"{min(float(dur or 6), 12.0):.2f}"
    lavfi = "[0:v][1:v]hstack=inputs=2:shortest=1[v];[2:v][v]vstack=inputs=2:shortest=1[o]"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", a, "-i", b, "-loop", "1", "-t", dur, "-r", "30", "-i", bar_p, "-filter_complex", lavfi,
                    "-map", "[o]", "-t", dur, "-c:v", "libx264", "-crf", "20", "-preset", "veryfast", "-pix_fmt", "yuv420p", out], check=False, timeout=120)

rows = []
for v in sorted(glob.glob("pairs/*_volc.mp4")):
    b = v.replace("_volc.mp4", "_bjxy.mp4")
    if not os.path.exists(b): continue
    base = os.path.basename(v)[:-9]; tag, ep = base.rsplit("_ep", 1)
    s10, p10 = metric(v, b, "ssim", 10), metric(v, b, "psnr", 10)
    sa, pa = metric(v, b, "ssim"), metric(v, b, "psnr")
    out = f"side_by_side/{tag}_ep{ep}_volcA100_vs_bjxy5090.mp4"; side_by_side(v, b, out)
    rows.append((tag, ep, s10, p10, sa, pa, os.path.getsize(v) // 1024, os.path.getsize(b) // 1024, out))
    print(tag, ep, s10, p10, sa, pa)
with open("quality_table.md", "w") as f:
    f.write("| pair | ep | SSIM (first 10 f) | PSNR dB (first 10 f) | SSIM (all) | PSNR (all) | volc KB | bjxy KB | video |\n|---|---|---|---|---|---|---|---|---|\n")
    for r in rows:
        f.write(f"| {r[0]} | {r[1]} | {r[2]:.4f} | {r[3]:.1f} | {r[4]:.4f} | {r[5]:.1f} | {r[6]} | {r[7]} | [mp4]({r[8]}) |\n")
import statistics as st
print("mean SSIM10", round(st.mean(r[2] for r in rows), 4), "mean PSNR10", round(st.mean(r[3] for r in rows), 1),
      "mean SSIM_all", round(st.mean(r[4] for r in rows), 4), "mean PSNR_all", round(st.mean(r[5] for r in rows), 1))
