import sys
sys.path.append('bev')
sys.path.append('proton')
sys.path.append('../bev')
sys.path.append('../proton')

from plan import Plan
import torch

from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm
import SimpleITK as sitk
import numpy as np
from torch.utils.data import Dataset, DataLoader


from proton_bev import rotate_image_batched
from geometry import *

def read_dose_parallel(file_paths, num_threads=32, scale_factor=1e5):
    n_files = len(file_paths)

    first_img = sitk.ReadImage(file_paths[0])
    spatial_shape = sitk.GetArrayFromImage(first_img).shape
    combined_array = np.zeros((n_files, *spatial_shape), dtype=np.float16)

    def load_and_insert(idx):
        file_path = file_paths[idx]
        image = sitk.ReadImage(file_path)
        # 2. Overwrite directly into the pre-allocated array
        combined_array[idx] = (sitk.GetArrayFromImage(image) * scale_factor).astype(np.float16)

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        list(tqdm(executor.map(load_and_insert, range(n_files)), total=n_files, desc="Reading files"))

    print(f"Starting parallel load with {num_threads} threads...")
    print(f"\nFinished. Final array shape: {combined_array.shape}")
    print(f"Data type: {combined_array.dtype}")

    return combined_array

class BeamData(Dataset):
    def __init__(self, plan):
        self.plan = plan
        self.rot_input_tensor = torch.randn((30,)+self.plan.img.GetSize()[::-1]).to(torch.float16).cuda()
        self.img = torch.tensor(plan.img_arr).cuda()

        # Beam specific information
        self.isocentre = self.plan.isocentre
        self.n_cp = self.plan.beam_info[self.plan.beam_id]['n_cp']
        self.sad = self.plan.beam_info[self.plan.beam_id]['sad']
        self.cp_info = self.plan.cp[self.plan.beam_id]
        self.angles = torch.tensor([self.cp_info[i]['ga'] for i in range(self.n_cp)]).cuda()


        self.dose = self.load_doses()
        self.rotate_dose()
        self.img_rot = self.rotate_img()
        self.bev = self.cal_bev_parallel()

    def load_doses(self):
        
        fl = [f'{self.plan.dose_dir}/Dose_B{self.plan.beam_id}_CP{i:03d}.mha' for i in range(self.n_cp)]
        dose = read_dose_parallel(fl, num_threads=32)
        dose = torch.tensor(dose)
        return dose
    

    def rotate_dose(self):
        
        for i in range(6):
            self.rot_input_tensor[:] = self.dose[30*i:(i+1)*30]
            d_rot = rotate_image_batched(
                self.rot_input_tensor,  # Shape: (D, H, W) or (1, 1, D, H, W) on GPU
                self.plan.img.GetSpacing(),  # Can now be a torch.Tensor or list/tuple
                self.plan.img.GetOrigin(),  # Can now be a torch.Tensor or list/tuple
                self.plan.isocentre,  # Can now be a torch.Tensor or list/tuple
                -self.angles[i*30:(i+1)*30],  # Can now be a torch.Tensor of angles on GPU
                axis="z",
                bg_value=0,
                pad_voxels=None,
            ).cpu()

            self.dose[30*i:(i+1)*30] = d_rot
            print('Dose rotated:', 30*i, (i+1)*30)

    def rotate_img(self):
        rots = []
        for i in range(6):
            rot = rotate_image_batched(
                self.img,  # Shape: (D, H, W) or (1, 1, D, H, W) on GPU
                self.plan.img.GetSpacing(),  # Can now be a torch.Tensor or list/tuple
                self.plan.img.GetOrigin(),  # Can now be a torch.Tensor or list/tuple
                self.plan.isocentre,  # Can now be a torch.Tensor or list/tuple
                -self.angles[i*30:(i+1)*30],  # Can now be a torch.Tensor of angles on GPU
                axis="z",
                bg_value=-1024,
                pad_voxels=None,
            ).cpu()
            rots.append(rot)
        return torch.cat(rots, dim=0)

    def cal_bev_parallel(self, num_threads=32):
        combined_array = np.zeros((self.n_cp, *self.img.shape), dtype=np.uint8)
        drawer = MLCDrawer(
                ref_img=self.plan.img, 
                isocentre=self.isocentre, 
                sad=self.sad
            )
            
        def cal_bev(idx):
            cp = self.cp_info[idx]

            bev = drawer.cal_bev_beam_path(
                mlc = MLC.get_mlc_segs_mm(cp["l"], cp["r"], self.isocentre)
            )
            
            combined_array[idx] = bev

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            list(tqdm(executor.map(cal_bev, range(self.n_cp)), total=self.n_cp, desc="Getting BEV"))

        print(f"Starting parallel load with {num_threads} threads...")
        print(f"\nFinished. Final array shape: {combined_array.shape}")
        print(f"Data type: {combined_array.dtype}")

        return combined_array 
        
    def __len__(self):
        return self.n_cp

    def __getitem__(self, idx):
        
        return self.doses[idx]

if __name__ == '__main__':
    pat_dir = '/workspace/DoseRAD2026Dev/data/DoseRAD2026/photon/training/1ABB006'
    plan = Plan(
        img_file_path=rf"{pat_dir}/image/ct.mha",
        info_json_path=rf"{pat_dir}/1ABB006.json",
        dose_dir=rf"{pat_dir}/dose",
    )
    print(plan.beam_info)

    d = BeamData(plan)
        






