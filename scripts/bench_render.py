"""Isaac Sim render / physics micro-benchmark on the real PsiBot grasp scene.

Loads the same assets the evaluation uses (lab room USD, PsiBot robot USD with its three RGB camera prims, WillowTable,
100 ml glass beaker) with Isaac Lab, then times three things separately:
  A) render-only : sim.render() + TiledCamera.update() for the 3 cameras (no physics step)
  B) physics-only: sim.step(render=False) at dt = 1/120 (PhysX GPU, TGS solver like the eval)
  C) eval cadence: 4 physics substeps + 1 render of the 3 cameras (decimation 4 -> 30 Hz control), i.e. one
     policy control step *without* policy inference or video encoding.
Reports ms per frame/step (median, p10, p90) after warm-up. Run with the eval's experience file and env vars.
Usage (inside the chembench_isaacsim51 env):
  python bench_render.py --headless --enable_cameras --device cuda:0 --experience <kit> [--frames 300] [--cam_w 640 --cam_h 480] [--cams 3]
"""
import argparse, json, os, statistics as st, sys, time
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--frames", type=int, default=300)
parser.add_argument("--cam_w", type=int, default=640)  # camera resolution (not --width/--height: those belong to AppLauncher)
parser.add_argument("--cam_h", type=int, default=480)
parser.add_argument("--cams", type=int, default=3)
parser.add_argument("--out", type=str, default="")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext, PhysxCfg
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from psilab import PSILAB_USD_ASSET_DIR

ROOM = PSILAB_USD_ASSET_DIR + "/Chemistrylab/lab_simple_1.25.usd"
ROBOT = PSILAB_USD_ASSET_DIR + "/robots/psibot/psibot_normal.usd"
TABLE = PSILAB_USD_ASSET_DIR + "/rigid_objects/high_poly/workbench/willow_table/WillowTable.usd"
BEAKER = PSILAB_USD_ASSET_DIR + "/asset_collection/sim_ready/solid_assets_all/glass_beaker_100ml/Beaker003.usd"
CAM_PRIMS = ["/World/Robot/camera_head_base/camera_head_color", "/World/Robot/camera_third_base/camera_third_color",
             "/World/Robot/camera_chest_base/camera_chest_color"][: args.cams]

sim_cfg = SimulationCfg(dt=1 / 120, render_interval=4, device=args.device,
                        physx=PhysxCfg(solver_type=1, max_position_iteration_count=32, max_velocity_iteration_count=4,
                                       bounce_threshold_velocity=0.002, enable_ccd=True))
sim = SimulationContext(sim_cfg)
sim_utils.UsdFileCfg(usd_path=ROOM).func("/World/Room", sim_utils.UsdFileCfg(usd_path=ROOM))
sim_utils.DomeLightCfg(intensity=1500.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=1500.0))
if os.path.exists(TABLE):
    sim_utils.UsdFileCfg(usd_path=TABLE).func("/World/Table", sim_utils.UsdFileCfg(usd_path=TABLE), translation=(0.0, 0.0, 0.0))
# Robot and beaker are spawned as plain USD prims (PhysX still simulates the articulation / rigid body); the Isaac Lab
# Articulation/RigidObject tensor wrappers are not needed for timing and their init failed on the shared nodes.
sim_utils.UsdFileCfg(usd_path=ROBOT).func("/World/Robot", sim_utils.UsdFileCfg(usd_path=ROBOT), translation=(0.0, 0.0, 0.0))
if os.path.exists(BEAKER):
    sim_utils.UsdFileCfg(usd_path=BEAKER).func("/World/Beaker", sim_utils.UsdFileCfg(usd_path=BEAKER), translation=(0.6, 0.0, 0.9))
cams = []
for p in CAM_PRIMS:
    try:
        cams.append(TiledCamera(TiledCameraCfg(prim_path=p, data_types=["rgb"], width=args.cam_w, height=args.cam_h, spawn=None)))
    except Exception as e:
        print(f"[bench] camera prim {p} not usable: {e}", flush=True)
sim.reset()
for _ in range(30):  # warm-up: shader compilation, PhysX GPU buffers
    sim.step(render=True)
    for c in cams: c.update(sim.get_physics_dt())
torch.cuda.synchronize()

def timeit(fn, n):
    xs = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); torch.cuda.synchronize(); xs.append((time.perf_counter() - t0) * 1000.0)
    xs.sort()
    return {"median_ms": round(st.median(xs), 2), "p10_ms": round(xs[len(xs) // 10], 2), "p90_ms": round(xs[9 * len(xs) // 10], 2), "n": n}

def render_only():
    sim.render()
    for c in cams: c.update(0.0)
    if cams: _ = cams[0].data.output["rgb"]

def physics_only():
    sim.step(render=False)

def eval_cadence():
    for _ in range(4): sim.step(render=False)
    sim.render()
    for c in cams: c.update(sim.get_physics_dt() * 4)
    if cams: _ = cams[0].data.output["rgb"]

res = {"gpu": torch.cuda.get_device_name(0), "cams": len(cams), "res": f"{args.cam_w}x{args.cam_h}", "concurrency_note": os.environ.get("BENCH_META", ""), "frames": args.frames,
       "render_only_per_frame": timeit(render_only, args.frames),
       "physics_only_per_step_dt1_120": timeit(physics_only, args.frames * 4),
       "eval_cadence_per_control_step": timeit(eval_cadence, args.frames)}
res["derived"] = {"render_fps_3cam": round(1000.0 / res["render_only_per_frame"]["median_ms"], 1),
                  "physics_steps_per_s": round(1000.0 / res["physics_only_per_step_dt1_120"]["median_ms"], 0),
                  "control_steps_per_s_no_policy": round(1000.0 / res["eval_cadence_per_control_step"]["median_ms"], 1)}
print("[bench] RESULT " + json.dumps(res), flush=True)
if args.out:
    with open(args.out, "w") as f: json.dump(res, f, indent=1)
simulation_app.close()
