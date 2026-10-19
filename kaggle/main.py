# Body of the Kaggle kernel; scripts/kaggle_build.py puts FILES and ARGS in front of it.
import base64
import subprocess
import sys
from pathlib import Path

WORK = Path("/kaggle/working")
PROJ = Path("/tmp/sun")
RAW = PROJ / "data" / "raw"
S3 = "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data"


def sh(cmd):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


for rel, b64 in FILES.items():  # noqa: F821
    f = PROJ / rel
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(base64.b64decode(b64))

sh("pip install -q transformers==5.20.1 pvlib datasets git+https://github.com/microsoft/MoGe.git")
RAW.mkdir(parents=True, exist_ok=True)
dates = sorted({d[:10] for d in DRIVES})  # noqa: F821
jobs = [(f"{S3}/{d}_calib.zip", RAW / f"{d}_calib.zip") for d in dates]
jobs += [(f"{S3}/{d}/{d}_sync.zip", RAW / f"{d}.zip") for d in DRIVES]  # noqa: F821
procs = [subprocess.Popen(f"curl -sL --retry 5 -o {dst} {url}", shell=True) for url, dst in jobs]
for p in procs:
    p.wait()
for d in dates:
    sh(f"cd {RAW} && unzip -q -o {d}_calib.zip && rm {d}_calib.zip")
for d in DRIVES:  # noqa: F821
    # colour camera and GPS/IMU only
    sh(f"cd {RAW} && unzip -q -o {d}.zip '*/image_02/*' '*/oxts/*' && rm {d}.zip")
sh(f"cd {PROJ} && {sys.executable} scripts/shadow_study.py --raw {RAW} --out {WORK}/study "
   + " ".join(ARGS) + " --drives " + " ".join(DRIVES))  # noqa: F821
