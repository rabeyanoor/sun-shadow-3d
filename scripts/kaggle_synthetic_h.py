"""H ablation of the random-street benchmark on held-out seeds, on a Kaggle CPU kernel.

    python scripts/kaggle_synthetic_h.py <kaggle-user>
    kaggle kernels push -p kaggle/build/sun-shadow-synth-h
    kaggle kernels output <user>/sun-shadow-synth-h -p outputs/kaggle_synth_h

Runs scripts/synthetic_benchmark.py on seeds 10000+ (never used elsewhere) for H = 10, 20, 40 m,
full protocol and focal grid, and writes synthetic_benchmark_h*.csv / synthetic_focal_grid_h*.csv.
"""
import base64
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
user = sys.argv[1]
files = {str(p.relative_to(ROOT)): base64.b64encode(p.read_bytes()).decode()
         for p in sorted([*ROOT.glob("src/*.py"), ROOT / "scripts/synthetic_benchmark.py"])}
body = '''
import base64, os, subprocess, sys
from pathlib import Path
PROJ, OUT = Path("/tmp/sun"), Path("/kaggle/working")
for rel, b64 in FILES.items():
    f = PROJ / rel
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(base64.b64decode(b64))
for h in (10, 20, 40):
    cmd = (f"cd {PROJ} && {sys.executable} scripts/synthetic_benchmark.py --scenes 200 --seed0 10000 --h-max {h} "
           f"--tag _h{h} --out {OUT} --workers {os.cpu_count()}")
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)
'''
out = ROOT / "kaggle" / "build" / "sun-shadow-synth-h"
out.mkdir(parents=True, exist_ok=True)
(out / "kernel.py").write_text(f"FILES = {json.dumps(files)}\n" + body)
(out / "kernel-metadata.json").write_text(json.dumps({
    "id": f"{user}/sun-shadow-synth-h", "title": "sun-shadow-synth-h", "code_file": "kernel.py", "language": "python",
    "kernel_type": "script", "is_private": True, "enable_gpu": False, "enable_internet": True}, indent=2))
print("built", out)
