from mlp import MLPProcessor, DoseModel, add_glow
from data import BeamData, Plan
import numpy as np
from pathlib import Path
import torch

model = DoseModel()
model.load_weight('photon/test_weights')

data_dir = Path("/workspace/DoseRAD2026Dev/data/DoseRAD2026/photon/training")

np.random.seed(1234)
pt_list = [f.name for f in data_dir.iterdir() if f.is_dir()]
np.random.shuffle(pt_list)
pt_tr = pt_list[:50]
pt_vl = pt_list[50:60]
pt_ts = pt_list[60:]


pt = pt_vl[2]
pat_dir = data_dir / pt

# Connet with the BeamData
plan = Plan(
    img_file_path=rf"{pat_dir}/image/ct.mha",
    info_json_path=rf"{pat_dir}/{pt}.json",
    dose_dir=rf"{pat_dir}/dose",
)
d = BeamData(plan, 10)

import time
a = time.time()
with torch.inference_mode():
    for img, bev, mask, loc, dose in d:
        print(1)
        proc = MLPProcessor(img, bev, dose)
        pred = proc.inference(model).clip(min=0)
        pred = add_glow(pred, bev)
b = time.time()
print(b-a)

# PID: 112612