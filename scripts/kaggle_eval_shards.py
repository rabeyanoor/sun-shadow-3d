"""Split the KITTI evaluation over several Kaggle CPU kernels.

    python scripts/kaggle_eval_shards.py <kaggle-user> 5 2 3 4 [--h=10]
    for i in 2 3 4; do kaggle kernels push -p kaggle/build/sun-shadow-eval-s$i; done
    kaggle kernels output <user>/sun-shadow-eval-s2 -p outputs/kaggle_shards/s2   # when finished

Kernel i scores the frames i, i+n, i+2n, ... of the cache (output of <user>/sun-shadow-cache)
and writes one CSV per frame to /kaggle/working/parts_h20; copy those into
outputs/study/parts_h20 and rerun shadow_eval.py with --parts to merge.
"""
import base64
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
args = [a for a in sys.argv[1:] if not a.startswith("--h=")]
H = float(next((a[4:] for a in sys.argv[1:] if a.startswith("--h=")), 20))
user, n, shards = args[0], int(args[1]), [int(x) for x in args[2:]]
tag = "" if H == 20 else f"-h{H:g}"
files = {str(p.relative_to(ROOT)): base64.b64encode(p.read_bytes()).decode()
         for p in sorted([ROOT / "src/observability.py", ROOT / "src/shadows.py", ROOT / "src/corruptions.py",
                          ROOT / "scripts/shadow_eval.py"])}
body = '''
import base64, os, subprocess, sys
from pathlib import Path
PROJ = Path("/tmp/sun")
for rel, b64 in FILES.items():
    f = PROJ / rel
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(base64.b64decode(b64))
def sh(cmd):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)
tar = subprocess.run("find /kaggle/input -name cache.tar | head -1", shell=True, capture_output=True, text=True).stdout.strip()
sh(f"cd /tmp && tar xf {tar}")
sh(f"cd {PROJ} && {sys.executable} scripts/shadow_eval.py --cache /tmp/cache --h-max {H_MAX} --workers {os.cpu_count()} "
   f"--shard {SHARD} --parts /kaggle/working/parts_h20 --out /kaggle/working/kitti_eval_shard.csv")
'''
for i in shards:
    name = f"sun-shadow-eval{tag}-s{i}"
    out = ROOT / "kaggle" / "build" / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "kernel.py").write_text(f"FILES = {json.dumps(files)}\nSHARD = {json.dumps(f'{i}/{n}')}\nH_MAX = {H:g}\n" + body)
    (out / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{user}/{name}", "title": name, "code_file": "kernel.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": False, "enable_internet": False,
        "kernel_sources": [f"{user}/sun-shadow-cache"]}, indent=2))
    print("built", out)
