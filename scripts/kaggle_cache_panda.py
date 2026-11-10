"""Pack src/ + scripts/cache_pandaset.py into one Kaggle GPU kernel that caches PandaSet frames.

Generated from scripts/kaggle_cache.py (KITTI).

    python scripts/kaggle_cache_panda.py <kaggle-user> [smoke]
    kaggle kernels push -p kaggle/build/sun-shadow-cache-panda

The ISTD shadow detector is reused from the output of the <user>/sun-shadow-study kernel. The
kernel writes one cache.tar with an .npz per frame and frames.csv; scripts/shadow_eval.py reads it.
"""
import base64
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
user = sys.argv[1]
smoke = len(sys.argv) > 2 and sys.argv[2] == "smoke"

DRIVES = []
ARGS = ["--per-seq 8", "--min-elevation 2"]
if smoke:
    DRIVES, ARGS = ["080", "001"], ["--per-seq 2"]

files = {str(p.relative_to(ROOT)): base64.b64encode(p.read_bytes()).decode()
         for p in sorted([*ROOT.glob("src/*.py"), ROOT / "scripts/cache_pandaset.py"])}
body = '''
import base64, subprocess, sys
from pathlib import Path
PROJ = Path("/tmp/sun")
def sh(cmd):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)
for rel, b64 in FILES.items():
    f = PROJ / rel
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(base64.b64decode(b64))
sh("pip install -q transformers==5.20.1 pvlib remotezip git+https://github.com/microsoft/MoGe.git")
weights = subprocess.run("find /kaggle/input -name 'shadow_net_*.pt' | head -1", shell=True,
                         capture_output=True, text=True).stdout.strip()
print("detector weights:", weights, flush=True)
root = subprocess.run("find /kaggle/input -path '*/001/meta/gps.json' | head -1", shell=True,
                      capture_output=True, text=True).stdout.strip()
root = str(Path(root).parents[2])  # .../001/meta/gps.json -> folder of the sequences
print("pandaset root:", root, flush=True)
sh(f"cd {PROJ} && {sys.executable} scripts/cache_pandaset.py --root {root} --weights {weights} --out /tmp/cache "
   + " ".join(ARGS) + ((" --seqs " + " ".join(DRIVES)) if DRIVES else ""))
sh("cd /tmp && tar cf /kaggle/working/cache.tar cache")
'''
name = "sun-shadow-cache-panda" + ("-smoke" if smoke else "")
out = ROOT / "kaggle" / "build" / name
out.mkdir(parents=True, exist_ok=True)
head = f"FILES = {json.dumps(files)}\nDRIVES = {json.dumps(DRIVES)}\nARGS = {json.dumps(ARGS)}\n"
(out / "kernel.py").write_text(head + body)
(out / "kernel-metadata.json").write_text(json.dumps({
    "id": f"{user}/{name}", "title": name, "code_file": "kernel.py", "language": "python",
    "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
    "kernel_sources": [f"{user}/sun-shadow-study"], "dataset_sources": ["pz19930809/pandaset"]}, indent=2))
print("built", out)
