"""Run the CPU-heavy scoring on a Kaggle TPU VM (96 CPU cores; no accelerator is used).

    python scripts/kaggle_eval.py <kaggle-user>
    kaggle kernels push -p kaggle/build/sun-shadow-eval

Reads cache.tar from the output of the <user>/sun-shadow-cache kernel, then runs
    the h_max ablation of the synthetic benchmark on held-out seeds (10000+),
    the synthetic benchmark and the KITTI evaluation for each h_max,
and writes every CSV to /kaggle/working/study.
"""
import base64
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
user = sys.argv[1]
H_MAX = [10, 20, 40]
files = {str(p.relative_to(ROOT)): base64.b64encode(p.read_bytes()).decode()
         for p in sorted([*ROOT.glob("src/*.py"), ROOT / "scripts/shadow_eval.py", ROOT / "scripts/synthetic_benchmark.py"])}
body = '''
import base64, os, subprocess, sys
from pathlib import Path
PROJ, OUT = Path("/tmp/sun"), Path("/kaggle/working/study")
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "ablation").mkdir(exist_ok=True)
def sh(cmd):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)
for rel, b64 in FILES.items():
    f = PROJ / rel
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(base64.b64decode(b64))
sh("pip install -q pvlib scipy pandas")
tar = subprocess.run("find /kaggle/input -name cache.tar | head -1", shell=True, capture_output=True, text=True).stdout.strip()
sh(f"cd /tmp && tar xf {tar}")
n = max(4, os.cpu_count() - 2)
py = sys.executable
for h in H_MAX:
    sh(f"cd {PROJ} && {py} scripts/synthetic_benchmark.py --scenes 150 --seed0 10000 --h-max {h} --tag _h{h} "
       f"--out {OUT}/ablation --workers {n}")
for h in H_MAX:
    sh(f"cd {PROJ} && {py} scripts/synthetic_benchmark.py --scenes 400 --h-max {h} --tag _h{h} --out {OUT} --workers {n}")
    sh(f"cd {PROJ} && {py} scripts/shadow_eval.py --cache /tmp/cache --h-max {h} --workers {n} --out {OUT}/kitti_eval_h{h}.csv")
sh(f"cp /tmp/cache/frames.csv {OUT}/kitti_frames.csv")
'''
name = "sun-shadow-eval"
out = ROOT / "kaggle" / "build" / name
out.mkdir(parents=True, exist_ok=True)
(out / "kernel.py").write_text(f"FILES = {json.dumps(files)}\nH_MAX = {json.dumps(H_MAX)}\n" + body)
(out / "kernel-metadata.json").write_text(json.dumps({
    "id": f"{user}/{name}", "title": name, "code_file": "kernel.py", "language": "python",
    "kernel_type": "script", "is_private": True, "enable_gpu": False, "enable_internet": True,
    "enable_tpu": True, "machine_shape": "Tpu1VmV38", "kernel_sources": [f"{user}/sun-shadow-cache"]}, indent=2))
print("built", out)
