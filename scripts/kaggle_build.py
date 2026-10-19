"""Pack src/ + the study script into one Kaggle kernel.

    python scripts/kaggle_build.py <kaggle-user> [smoke]
    kaggle kernels push -p kaggle/build/<name>
"""
import base64
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
user = sys.argv[1]
smoke = len(sys.argv) > 2 and sys.argv[2] == "smoke"

DRIVES = ["2011_09_26_drive_0001", "2011_09_26_drive_0005", "2011_09_26_drive_0009",
          "2011_09_26_drive_0011", "2011_09_26_drive_0013"]
if smoke:
    # CPU kernel: a few ISTD images, one frame, the small model
    DRIVES, ARGS = DRIVES[:1], ["--shadow-epochs 1", "--istd-limit 16", "--max-frames 1", "--models dav2-s"]
else:
    ARGS = ["--shadow-epochs 8", "--every 10", "--max-frames 15"]

files = {str(p.relative_to(ROOT)): base64.b64encode(p.read_bytes()).decode()
         for p in sorted([*ROOT.glob("src/*.py"), ROOT / "scripts/shadow_study.py"])}
name = "sun-shadow-study" + ("-smoke" if smoke else "")
out = ROOT / "kaggle" / "build" / name
out.mkdir(parents=True, exist_ok=True)
head = f"FILES = {json.dumps(files)}\nDRIVES = {json.dumps(DRIVES)}\nARGS = {json.dumps(ARGS)}\n"
(out / "kernel.py").write_text(head + (ROOT / "kaggle" / "main.py").read_text())
(out / "kernel-metadata.json").write_text(json.dumps({
    "id": f"{user}/{name}", "title": name, "code_file": "kernel.py", "language": "python",
    "kernel_type": "script", "is_private": True, "enable_gpu": not smoke, "enable_internet": True}, indent=2))
print("built", out)
