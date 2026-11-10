"""Train shadow detector version 2 (SegFormer-B5, SBU + ISTD, low-sun augmentation) and compare it with v1.

    python scripts/train_detector.py --sbu /kaggle/input/sbu-shadow --old shadow_net_segformer_b2_istd.pt \
        --out detector_v2 [--epochs 10] [--limit 64]

Writes <out>/shadow_net_v2_segformer_b5_sbu_istd.pt and <out>/detector_ber.csv with the balanced error
rate (BER, %, lower is better) of both detectors on the ISTD and SBU test splits. Only these test splits
are looked at; the driving datasets are never used to train or choose the detector.
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import shadow_net  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--sbu", required=True, help="folder that contains SBU-Train/ and SBU-Test/")
p.add_argument("--old", required=True, help="state dict of detector v1 (mit-b2, ISTD, 384 px)")
p.add_argument("--out", required=True)
p.add_argument("--epochs", type=int, default=10)
p.add_argument("--limit", type=int, default=None, help="smoke test: use only this many samples per split")
a = p.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)

sbu_tr, sbu_te = shadow_net.load_sbu(a.sbu, "train"), shadow_net.load_sbu(a.sbu, "test")
istd_tr, istd_te = shadow_net.load_istd("train"), shadow_net.load_istd("test")
if a.limit:
    sbu_tr, sbu_te = sbu_tr[: a.limit], sbu_te[: a.limit]
    istd_tr, istd_te = istd_tr.select(range(a.limit)), istd_te.select(range(a.limit))
print(f"SBU train {len(sbu_tr)} test {len(sbu_te)}, ISTD train {len(istd_tr)} test {len(istd_te)}", flush=True)

t0 = time.time()
net = shadow_net.train_v2(shadow_net.Concat(sbu_tr, istd_tr), epochs=a.epochs,
                          log=lambda m: print(f"{m}  [{(time.time() - t0) / 60:.0f} min]", flush=True))
weights = out / "shadow_net_v2_segformer_b5_sbu_istd.pt"
torch.save(net.state_dict(), weights)

old = shadow_net.build("nvidia/mit-b2")
old.load_state_dict(torch.load(a.old, map_location=shadow_net.DEVICE))
old.eval()
rows = []
for name, d in [("ISTD", istd_te), ("SBU", sbu_te)]:
    rows.append(dict(test=name, detector="v1 (B2, ISTD)", ber=shadow_net.evaluate(old, data=d)))
    for tta in (False, True):
        rows.append(dict(test=name, detector="v2 (B5, SBU+ISTD)" + (" + flip" if tta else ""),
                         ber=shadow_net.evaluate(net, data=d, size=512, tta=tta)))
print(pd.DataFrame(rows).to_string(index=False), flush=True)
pd.DataFrame(rows).to_csv(out / "detector_ber.csv", index=False)
print("saved", weights, flush=True)
