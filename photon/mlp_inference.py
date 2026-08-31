from mlp import MLPProcessor, DoseModel, add_glow
from data import BeamData, Plan
import numpy as np
from pathlib import Path
import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


def infer_parallel(d, num_threads=8):
    n_cp = len(d)

    preds = {}

    def infer(idx):
        img, bev, mask, loc, dose = d[idx]
        with torch.inference_mode():
            proc = MLPProcessor(img, bev, dose)
            pred = proc.inference(model)
            pred = torch.clip(add_glow(pred, bev), min=0)
        preds[idx] = pred.detach().cpu()

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        list(
            tqdm(
                executor.map(infer, range(n_cp)),
                total=n_cp,
                desc="Inferencing",
            )
        )
        

    return preds


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
d = BeamData(plan, 180)

import time
a = time.time()
infer_parallel(d)
b = time.time()
print(b-a)

# PID: 112612