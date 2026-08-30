import torch
from scipy import ndimage
import numpy as np
from scipy import ndimage
from scipy.ndimage import distance_transform_edt
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class MLPProcessor():
    def __init__(self, ct, bev, dose):
        self.ct = ct.clip(min=-1024)
        self.bev = bev
        self.dose = dose

        self.dim = 1
        self.ct_norm = self.ct/1000+1.024
        self.mask = self.ct > -1024

        self.depth = torch.cumsum(self.mask, dim=self.dim)
        self.wed = torch.cumsum(self.mask*self.ct_norm, dim=self.dim)
        self.red = self.cal_red()
        self.rings = self.cal_onion()
        self.keys = ['bev3', 'ring3', 'ring2', 'ring1']

        self.data = torch.stack([self.ct_norm, self.bev, self.depth, self.wed, self.red], dim=0).float()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def cal_red(self):
        red = torch.zeros_like(self.ct)
        red[self.ct<0] = 1 + 0.001*self.ct[self.ct<0]
        red[self.ct>=0] = 1 + 0.005*self.ct[self.ct>=0]
        red = torch.cumsum(self.mask*red, dim=self.dim)

        return red
    
    def cal_onion(self):
        bev3, bev2, bev1 = [torch.tensor(ndimage.binary_erosion(
            self.bev.numpy(), structure=None, iterations=i
        )).to(int) for i in (3,2,1)]

        ring1 = self.bev - bev1
        ring2 = bev1 - bev2
        ring3 = bev2 - bev3

        return dict(
            bev3 = bev3,
            ring3 = ring3,
            ring2 = ring2,
            ring1 = ring1
        )
    
    @staticmethod
    def sample_from_core(foreground_mask, to_sample):
        outside_mask = (1-foreground_mask).astype(np.uint8)
        distances, (nearest_z, nearest_y, nearest_x) = distance_transform_edt(
            outside_mask, return_indices=True
        )

        return to_sample[nearest_z, nearest_y, nearest_x], distances
    
    def prepare_ring_data(self, key, inner_dose=None, with_dose=False):

        ring_1d = self.rings[key].ravel()
        
        # Project the inner dose to the rest of volume
        if inner_dose is not None:
            border_dose, distances = self.sample_from_core((inner_dose!=0).numpy(), inner_dose)
            distances = torch.tensor(distances)
            inner_dose_data = torch.stack([border_dose, distances], dim=0).float()
            
            data = torch.cat([self.data, inner_dose_data], dim=0)
        else:
            data = self.data.clone()

        x = data.reshape((data.shape[0], -1)).moveaxis(1, 0)
        x = x[ring_1d!=0].to(self.device)

        if with_dose is True:
            y = self.dose.unsqueeze(0).reshape((1, -1)).moveaxis(1,0)
            y = y[ring_1d!=0].to(self.device)
            return x, y
        else:
            return x
        
    def get_xy(self):
        xs = []
        ys = []

        for key in self.keys:
            if key == 'bev3':
                x, y = self.prepare_ring_data(key, with_dose=True)
            else:
                # Create an inner dose using masked gt dose
                inner_dose = self.dose.clone()
                inner_dose[self.rings[key]==0] = 0
                x, y = self.prepare_ring_data(key, inner_dose=inner_dose, with_dose=True)

            xs.append(x)
            ys.append(y)
        return xs, ys
        
    
    def get_full_map(self, preds):

        pred_maps = [torch.zeros_like(self.dose) for i in range(4)]

        for pred_map, key, pred in zip(pred_maps, self.keys, preds):
            pred_map[self.rings[key]!=0] = pred.ravel()

        return torch.stack(pred_maps, dim=0).sum(0)



class SimpleMLP(nn.Module):
    def __init__(self, in_channel=5):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(in_channel, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.out = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        out = self.out(x)
        return out
    
class DoseModel():
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.bev3 = SimpleMLP(in_channel=5).to(self.device)
        self.ring3 = SimpleMLP(in_channel=7).to(self.device)
        self.ring2 = SimpleMLP(in_channel=7).to(self.device)
        self.ring1 = SimpleMLP(in_channel=7).to(self.device)

    def __call__(self, xs):
        models = [self.bev3, self.ring3, self.ring2, self.ring1]
        preds = [model(x) for model, x in zip(models, xs)]

        return preds
            
if __name__ == "__main__":
    input_tensor = torch.load('data.pt')
    ct, bev, mask, dose = input_tensor[:,0].unbind()
    ct = ct.moveaxis(0, 1)
    bev = bev.moveaxis(0, 1)
    mask = mask.moveaxis(0, 1)
    dose = dose.moveaxis(0, 1)


    proc = MLPProcessor(ct, bev, dose)
    model = DoseModel()

    # Training
    optimizers = [
        optim.Adam(getattr(model, key).parameters(), lr=0.001) for key in proc.keys
    ]
    l1 = nn.L1Loss()
    l2 = nn.MSELoss()

    print("Starting MLP training...")

    epochs = 1000
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

        # ls = [(((pred/(y+1e-5)).abs()-1)**2).max()*w for (pred,y,w) in zip(preds,ys,weights)]
        

        pbar.set_postfix_str(f'Epoch [{epoch+1}/{epochs}], Loss: {ls}')
