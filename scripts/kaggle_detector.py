"""One Kaggle GPU kernel: train detector v2, compare its BER with v1, and rerun it on all cached frames.

    python scripts/kaggle_detector.py <kaggle-user> [smoke]
    kaggle kernels push -p kaggle/build/sun-shadow-detector-v2
    kaggle kernels output <user>/sun-shadow-detector-v2 -p outputs/kaggle_detector_v2     # when finished

Output: detector_v2/{weights, detector_ber.csv} and prob2_kitti.tar / prob2_panda.tar (one .npy per frame),
read by scripts/shadow_eval.py --prob2 (observation "net2"). v1 weights come from <user>/sun-shadow-study,
the cached frame lists (and, for the check, the cached v1 probabilities) from the two cache kernels.
"""
import base64
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
user = sys.argv[1]
smoke = len(sys.argv) > 2 and sys.argv[2] == "smoke"
files = {str(p.relative_to(ROOT)): base64.b64encode(p.read_bytes()).decode()
         for p in sorted([*ROOT.glob("src/*.py"), ROOT / "scripts/train_detector.py", ROOT / "scripts/reprob.py"])}
body = '''
import base64, subprocess, sys
from pathlib import Path
PROJ = Path("/tmp/sun")
for rel, b64 in FILES.items():
    f = PROJ / rel
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(base64.b64decode(b64))
def sh(cmd):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)
def find(pattern):
    return subprocess.run(f"find /kaggle/input {pattern} | head -1", shell=True, capture_output=True,
                          text=True).stdout.strip()
sh("pip install -q transformers==5.20.1 remotezip datasets")
old = find("-name 'shadow_net_*.pt'")
sbu = str(Path(find("-type d -name 'SBU-Test'")).parent)
panda = str(Path(find("-path '*/001/meta/gps.json'")).parents[2])
print("v1:", old, "sbu:", sbu, "panda:", panda, flush=True)
W = "/kaggle/working"
sh(f"cd {PROJ} && {sys.executable} scripts/train_detector.py --sbu {sbu} --old {old} --out {W}/detector_v2 " + TRAIN)
v2 = f"{W}/detector_v2/shadow_net_v2_segformer_b5_sbu_istd.pt"
for data, tar in [("kitti", find("-path '*sun-shadow-cache/cache.tar'")),
                  ("panda", find("-path '*sun-shadow-cache-panda/cache.tar'"))]:
    sh(f"mkdir -p /tmp/{data} && cd /tmp/{data} && tar xf {tar}")
    cache = f"/tmp/{data}/cache"
    if LIMIT:
        sh(f"head -n {LIMIT + 1} {cache}/frames.csv > /tmp/{data}_frames.csv")
    else:
        sh(f"cp {cache}/frames.csv /tmp/{data}_frames.csv")
    sh(f"cd {PROJ} && {sys.executable} scripts/reprob.py --data {data} --root {panda} --frames /tmp/{data}_frames.csv "
       f"--weights {v2} --tta --out /tmp/prob2_{data} --check-weights {old} --check-cache {cache}")
    sh(f"cd /tmp && tar cf {W}/prob2_{data}.tar prob2_{data} && rm -rf /tmp/{data}")
'''
name = "sun-shadow-detector-v2" + ("-smoke" if smoke else "")
train = "--epochs 1 --limit 32" if smoke else "--epochs 10"
out = ROOT / "kaggle" / "build" / name
out.mkdir(parents=True, exist_ok=True)
head = f"FILES = {json.dumps(files)}\nTRAIN = {json.dumps(train)}\nLIMIT = {4 if smoke else 0}\n"
(out / "kernel.py").write_text(head + body)
(out / "kernel-metadata.json").write_text(json.dumps({
    "id": f"{user}/{name}", "title": name, "code_file": "kernel.py", "language": "python",
    "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
    "kernel_sources": [f"{user}/sun-shadow-study", f"{user}/sun-shadow-cache", f"{user}/sun-shadow-cache-panda"],
    "dataset_sources": ["pz19930809/pandaset", "sayantandas30011998/sbu-shadow"]}, indent=2))
print("built", out)
