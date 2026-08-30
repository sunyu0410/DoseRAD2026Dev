import torch
from scipy import ndimage
import torch.nn.functional as F
from data import BeamData, Plan
import torch.nn as nn
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import ndimage
from mlp import MLPProcessor, DoseModel
import torch.optim as optim
import gc

data_dir = Path("/workspace/DoseRAD2026Dev/data/DoseRAD2026/photon/training")

np.random.seed(1234)
pt_list = [f.name for f in data_dir.iterdir() if f.is_dir()]
np.random.shuffle(pt_list)

pt_tr = pt_list[:50]
pt_vl = pt_list[50:60]
pt_ts = pt_list[60:]

model = DoseModel()


l1 = nn.L1Loss()
l2 = nn.MSELoss()

n = -1

# Round 1
optimizers = [
    optim.Adam(getattr(model, key).parameters(), lr=0.001) for key in ['bev3', 'ring3', 'ring2', 'ring1']
]

for pt_id, pt in enumerate(pt_tr):
    gc.collect()
    torch.cuda.empty_cache()
    n += 1
    pat_dir = data_dir / pt
    plan = Plan(
        img_file_path=rf"{pat_dir}/image/ct.mha",
        info_json_path=rf"{pat_dir}/{pt}.json",
        dose_dir=rf"{pat_dir}/dose",
    )
    d = BeamData(plan, 5)
    for img, bev, mask, loc, dose in d:
        proc = MLPProcessor(img, bev, dose)

        epochs = 50
        pbar = tqdm(range(epochs))
        for epoch in pbar:
            # Forward pass
            xs, ys = proc.get_xy()

            preds = model(xs)
            weights = [3,1,1,1]

            ls = [l1(pred, y)*w for (pred,y,w) in zip(preds,ys,weights)]
            for o, l in zip(optimizers, ls):
                o.zero_grad()
                l.backward()
                o.step()


            pbar.set_postfix_str(f'Epoch [{epoch+1}/{epochs}], Loss: {torch.stack(ls).tolist()}')

    # Save weights
    print(n, n%10, n%10==0)
    if n%10 == 0:
        torch.save(model.bev3.state_dict(), f'mlp_weights/{n}-bev3-{pt_id}.pth')
        torch.save(model.ring3.state_dict(), f'mlp_weights/{n}-ring3-{pt_id}.pth')
        torch.save(model.ring2.state_dict(), f'mlp_weights/{n}-ring2-{pt_id}.pth')
        torch.save(model.ring1.state_dict(), f'mlp_weights/{n}-ring1-{pt_id}.pth')

# lr 1e-4
optimizers = [
    optim.Adam(getattr(model, key).parameters(), lr=0.0001) for key in ['bev3', 'ring3', 'ring2', 'ring1']
]

for pt_id, pt in enumerate(pt_tr):
    gc.collect()
    torch.cuda.empty_cache()
    n += 1
    pat_dir = data_dir / pt
    plan = Plan(
        img_file_path=rf"{pat_dir}/image/ct.mha",
        info_json_path=rf"{pat_dir}/{pt}.json",
        dose_dir=rf"{pat_dir}/dose",
    )
    d = BeamData(plan, 30)
    for img, bev, mask, loc, dose in d:
        proc = MLPProcessor(img, bev, dose)

        epochs = 50
        pbar = tqdm(range(epochs))
        for epoch in pbar:
            # Forward pass
            xs, ys = proc.get_xy()

            preds = model(xs)
            weights = [3,1,1,1]

            ls = [l1(pred, y)*w for (pred,y,w) in zip(preds,ys,weights)]
            for o, l in zip(optimizers, ls):
                o.zero_grad()
                l.backward()
                o.step()


            pbar.set_postfix_str(f'Epoch [{epoch+1}/{epochs}], Loss: {torch.stack(ls).tolist()}')

    # Save weights
    if n%10 == 0:
        torch.save(model.bev3.state_dict(), f'mlp_weights/{n}-bev3-{pt_id}.pth')
        torch.save(model.ring3.state_dict(), f'mlp_weights/{n}-ring3-{pt_id}.pth')
        torch.save(model.ring2.state_dict(), f'mlp_weights/{n}-ring2-{pt_id}.pth')
        torch.save(model.ring1.state_dict(), f'mlp_weights/{n}-ring1-{pt_id}.pth')

# lr 1e-5
optimizers = [
    optim.Adam(getattr(model, key).parameters(), lr=0.00001) for key in ['bev3', 'ring3', 'ring2', 'ring1']
]

for pt_id, pt in enumerate(pt_tr):
    gc.collect()
    torch.cuda.empty_cache()
    n += 1
    pat_dir = data_dir / pt
    plan = Plan(
        img_file_path=rf"{pat_dir}/image/ct.mha",
        info_json_path=rf"{pat_dir}/{pt}.json",
        dose_dir=rf"{pat_dir}/dose",
    )
    d = BeamData(plan, 60)
    for img, bev, mask, loc, dose in d:
        proc = MLPProcessor(img, bev, dose)

        epochs = 50
        pbar = tqdm(range(epochs))
        for epoch in pbar:
            # Forward pass
            xs, ys = proc.get_xy()

            preds = model(xs)
            weights = [3,1,1,1]

            ls = [l1(pred, y)*w for (pred,y,w) in zip(preds,ys,weights)]
            for o, l in zip(optimizers, ls):
                o.zero_grad()
                l.backward()
                o.step()


            pbar.set_postfix_str(f'Epoch [{epoch+1}/{epochs}], Loss: {torch.stack(ls).tolist()}')

    # Save weights
    if n%10 == 0:
        torch.save(model.bev3.state_dict(), f'mlp_weights/{n}-bev3-{pt_id}.pth')
        torch.save(model.ring3.state_dict(), f'mlp_weights/{n}-ring3-{pt_id}.pth')
        torch.save(model.ring2.state_dict(), f'mlp_weights/{n}-ring2-{pt_id}.pth')
        torch.save(model.ring1.state_dict(), f'mlp_weights/{n}-ring1-{pt_id}.pth')
