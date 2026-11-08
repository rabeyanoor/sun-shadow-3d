"""Pack src/ + scripts/cache_kitti.py into one Kaggle GPU kernel that caches the KITTI frames.

    python scripts/kaggle_cache.py <kaggle-user> [smoke]
    kaggle kernels push -p kaggle/build/sun-shadow-cache

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

# 26 Sep 2011 (city, residential, road) plus four later days at other times of day,
# so that the sun elevation and its direction in the camera vary
DRIVES = [f"2011_09_26_drive_{n:04d}" for n in (
    1, 5, 9, 11, 13, 14, 15, 18, 19, 22, 23, 27, 28, 29, 32, 35, 39, 46, 51, 56, 57, 59, 61, 64, 70, 84, 91, 93, 95,
    96, 104, 106, 117)]
DRIVES += ["2011_09_28_drive_0002", "2011_09_29_drive_0004", "2011_09_29_drive_0026", "2011_09_30_drive_0016",
           "2011_09_30_drive_0020", "2011_10_03_drive_0042", "2011_10_03_drive_0047"]
ARGS = ["--per-drive 15"]
if smoke:
    DRIVES, ARGS = ["2011_09_26_drive_0001", "2011_09_29_drive_0026"], ["--per-drive 2"]

files = {str(p.relative_to(ROOT)): base64.b64encode(p.read_bytes()).decode()
         for p in sorted([*ROOT.glob("src/*.py"), ROOT / "scripts/cache_kitti.py"])}
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
sh(f"cd {PROJ} && {sys.executable} scripts/cache_kitti.py --weights {weights} --out /tmp/cache "
   + " ".join(ARGS) + " --drives " + " ".join(DRIVES))
sh("cd /tmp && tar cf /kaggle/working/cache.tar cache")
'''
name = "sun-shadow-cache" + ("-smoke" if smoke else "")
out = ROOT / "kaggle" / "build" / name
out.mkdir(parents=True, exist_ok=True)
head = f"FILES = {json.dumps(files)}\nDRIVES = {json.dumps(DRIVES)}\nARGS = {json.dumps(ARGS)}\n"
(out / "kernel.py").write_text(head + body)
(out / "kernel-metadata.json").write_text(json.dumps({
    "id": f"{user}/{name}", "title": name, "code_file": "kernel.py", "language": "python",
    "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
    "kernel_sources": [f"{user}/sun-shadow-study"]}, indent=2))
print("built", out, len(DRIVES), "drives")
