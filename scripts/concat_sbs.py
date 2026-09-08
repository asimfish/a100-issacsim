"""Concatenate the 15 side-by-side clips into one video with a per-clip caption bar (pair, episode, SSIM/PSNR) and a
2-second title card. Output: media/A100_vs_5090_all_pairs.mp4 (+ copy to ~/Desktop/research/volc_render_check)."""
import os, re, subprocess, tempfile
from PIL import Image, ImageDraw, ImageFont
REPO = os.path.expanduser("~/Code/a100-issacsim"); os.chdir(REPO)
FONT = "/System/Library/Fonts/Helvetica.ttc"
def font(sz):
    try: return ImageFont.truetype(FONT, sz)
    except Exception: return ImageFont.load_default()
# quality numbers
q = {}
for l in open("results/quality_table.md"):
    c = [x.strip() for x in l.strip().strip("|").split("|")]
    if len(c) >= 6 and re.match(r"\d{3}$", c[1]): q[(c[0], c[1])] = (c[2], c[3])
names = {"alcohol_lamp16k_s3952": "alcohol_lamp  ACT relabel 16k  s3952", "brown_vol16k_s3951": "brown_volumetric_flask_250ml  ACT relabel 16k  s3951",
         "clear_vol16k_s3951": "clear_volumetric_flask_250ml  ACT relabel 16k  s3951", "crbL16k_s3951": "clear_reagent_bottle_large  ACT relabel 16k  s3951",
         "erlenmeyer_ao_s3961": "erlenmeyer_flask  ACT action-only relabel 8k  s3961"}
tmp = tempfile.mkdtemp(); parts = []
def bar(text, w=1280, h=36, fg=(255, 230, 80), bg=(20, 20, 20), sz=20):
    im = Image.new("RGB", (w, h), bg); ImageDraw.Draw(im).text((12, 8), text, fill=fg, font=font(sz)); return im
# title card
card = Image.new("RGB", (1280, 508 + 36), (15, 15, 15)); d = ImageDraw.Draw(card)
lines = ["Isaac Sim 5.1 headless RTX rendering:  A100 (no RT cores)  vs  RTX 5090", "",
         "Same checkpoint, same episode index, same eval seed (42) -> identical initial layout.", "Left: volc A100-80G (Vulkan, ray tracing on CUDA cores, no DLSS).   Right: bjxy RTX 5090 (RT cores).",
         "15 clips = 5 checkpoints x episodes 1-3, played at 0.5x speed.  Caption shows SSIM / PSNR over the first 10 frames.", "", "github.com/asimfish/a100-issacsim   -   2026-09-08"]
y = 140
for i, t in enumerate(lines):
    d.text((60, y), t, fill=(255, 255, 255) if i == 0 else (200, 200, 200), font=font(34 if i == 0 else 22)); y += 52 if i == 0 else 34
cp = os.path.join(tmp, "title.png"); card.save(cp)
tp = os.path.join(tmp, "title.mp4")
subprocess.run(["ffmpeg", "-v", "error", "-y", "-loop", "1", "-t", "3", "-r", "30", "-i", cp, "-c:v", "libx264", "-crf", "20", "-preset", "veryfast", "-pix_fmt", "yuv420p", tp], check=True)
parts.append(tp)
clips = sorted(f for f in os.listdir("media/side_by_side") if f.endswith(".mp4"))
for i, f in enumerate(clips):
    m = re.match(r"(.+)_ep(\d{3})_volcA100_vs_bjxy5090\.mp4", f); tag, ep = m.group(1), m.group(2)
    s, p = q.get((tag, ep), ("?", "?"))
    cap = f"[{i+1}/{len(clips)}]  {names.get(tag, tag)}    -    episode {int(ep)}    -    SSIM {s}  /  PSNR {p} dB  (first 10 frames)"
    bp = os.path.join(tmp, f"bar{i}.png"); bar(cap).save(bp)
    out = os.path.join(tmp, f"part{i:02d}.mp4")
    dur = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", f"media/side_by_side/{f}"], capture_output=True, text=True).stdout.strip()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f"media/side_by_side/{f}", "-loop", "1", "-t", dur, "-r", "30", "-i", bp,
                    "-filter_complex", "[0:v][1:v]vstack=inputs=2:shortest=1,setpts=2.0*PTS[o]", "-map", "[o]",
                    "-c:v", "libx264", "-crf", "20", "-preset", "veryfast", "-pix_fmt", "yuv420p", out], check=True, timeout=180)
    parts.append(out)
lst = os.path.join(tmp, "list.txt"); open(lst, "w").write("".join(f"file '{p}'\n" for p in parts))
out = "media/A100_vs_5090_all_pairs.mp4"
subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c:v", "libx264", "-crf", "20", "-preset", "veryfast", "-pix_fmt", "yuv420p", out], check=True, timeout=600)
print(out, os.path.getsize(out) // 1024, "KB", subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", out], capture_output=True, text=True).stdout.strip(), "s")
