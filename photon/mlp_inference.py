from mlp import MLPProcessor, DoseModel, add_glow
from data import BeamData, Plan
import numpy as np
from pathlib import Path
import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import json

def infer_parallel(d, num_threads=8):
    n_cp = len(d)

    preds = {}
    doses = {}

    def infer(idx):
        img, bev, mask, loc, dose = d[idx]
        with torch.inference_mode():
            proc = MLPProcessor(img, bev, dose)
            pred = proc.inference(model)
            pred = torch.clip(add_glow(pred, bev), min=0)
        preds[idx] = pred.detach().cpu()
        doses[idx] = dose

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        list(
            tqdm(
                executor.map(infer, range(n_cp)),
                total=n_cp,
                desc="Inferencing",
            )
        )
        

    return preds, doses


model = DoseModel()
model.load_weight('photon/test_weights')

weight_dir = Path('photon/mlp_weights')
data_dir = Path("data/DoseRAD2026/photon/training")

def update(model, n):
    fnames = list(weight_dir.glob(f'{n}-*'))
    fnames.sort()

    model.bev3.load_state_dict(torch.load(fnames[0]))
    model.ring1.load_state_dict(torch.load(fnames[1]))
    model.ring2.load_state_dict(torch.load(fnames[2]))
    model.ring3.load_state_dict(torch.load(fnames[3]))

    print('Weights updated')


np.random.seed(1234)
pt_list = [f.name for f in data_dir.iterdir() if f.is_dir()]
np.random.shuffle(pt_list)
pt_tr = pt_list[:50]
pt_vl = pt_list[50:60]
pt_ts = pt_list[60:]

from torch.nn import L1Loss
loss = L1Loss()

for pt in pt_vl:
    pat_dir = data_dir / pt

    # Connet with the BeamData
    plan = Plan(
        img_file_path=rf"{pat_dir}/image/ct.mha",
        info_json_path=rf"{pat_dir}/{pt}.json",
        dose_dir=rf"{pat_dir}/dose",
    )
    d = BeamData(plan, 180)

    for n in range(0, 123, 2):

        update(model, n)
        preds, ys = infer_parallel(d)

        for i in range(180):
            l1 = loss(preds[i], ys[i])
            top = ys[i]>(ys[i].max()*0.9)
            rel = preds[i][top]/ys[i][top]

            row = dict(
                pt = pt,
                n = n,
                l1_max = l1.max().item(),
                l1_min = l1.min().item(),
                l1_mean = l1.mean().item(),
                rel_max = rel.max().item(),
                rel_min = rel.min().item(),
                rel_mean = rel.mean().item()
            )

            json.dump(row, open(f'photon/val_result/{pt}-{n}.json', 'w'))

        