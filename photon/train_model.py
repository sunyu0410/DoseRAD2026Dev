import torch
from scipy import ndimage
import torch.nn.functional as F
from data import BeamData, Plan
from models import BeamNet, LaplacianSmoothness2DLoss
import torch.nn as nn
from pathlib import Path
import numpy as np

model = BeamNet()
loss = nn.MSELoss()  # Standard for continuous radiation dose matching
smooth_loss = LaplacianSmoothness2DLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-1, weight_decay=1e-2)


data_dir = Path("/workspace/DoseRAD2026Dev/data/DoseRAD2026/photon/training")

np.random.seed(1234)

pt_list = [f.name for f in data_dir.iterdir() if f.is_dir()]
np.random.shuffle(pt_list)

pt_tr = pt_list[:50]
pt_vl = pt_list[50:60]
pt_ts = pt_list[60:]

for pt in pt_tr:
    pat_dir = data_dir / pt
    plan = Plan(
        img_file_path=rf"{pat_dir}/image/ct.mha",
        info_json_path=rf"{pat_dir}/{pt}.json",
        dose_dir=rf"{pat_dir}/dose",
    )
    print(pt)

    for beam_id in range(plan.n_beams):
        plan.set_state(beam_id)
        d = BeamData(plan, 30)
    # for i in range(1000):
        for data in d:

            img, bev, mask, dose = data.unsqueeze(2).unbind(dim=0)

            mask1 = torch.tensor(ndimage.binary_dilation(bev.numpy(), structure=None, iterations=3)).to(int)
            ring1 = (mask1 - bev).bool().cuda()
            mask1 = mask1.bool().cuda()
            bev = bev.bool().cuda()
            img = img.cuda()
            y = dose.cuda()

            pred = model(img, bev)

            l1 = loss(pred[bev], y[bev])
            l2 = loss(pred[ring1], y[ring1])
            l3 = loss(pred[~mask1], y[~mask1])
            l4 = smooth_loss(pred, y)
            l = l1*2 + l2 + l3 + l4

            l.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(l.item())