from bev.plan import Plan
import torch

from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm
import SimpleITK as sitk
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from proton.proton_bev import rotate_image_batched
from bev.geometry import *
from bev.bev_torch import make_grid_5d, get_bev_torch, mm2idx, cal_scales, draw_iso_mlc

def read_dose_parallel(file_paths, num_threads=8, scale_factor=1e5):
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
        self.img = torch.tensor(plan.img_arr).to(torch.float16).cuda()

        # Beam specific information
        self.isocentre = self.plan.isocentre
        self.n_cp = self.plan.beam_info[self.plan.beam_id]['n_cp']
        self.sad = self.plan.beam_info[self.plan.beam_id]['sad']
        self.cp_info = self.plan.cp[self.plan.beam_id]
        self.angles = torch.tensor([self.cp_info[i]['ga'] for i in range(self.n_cp)]).cuda()

        self.dose = self.load_doses()
        self.rotate_dose()
        self.img_rot = self.rotate_img()
        # self.bev = self.cal_bev_parallel()
        self.bev = self.cal_bev()

    def load_doses(self):
        
        fl = [f'{self.plan.dose_dir}/Dose_B{self.plan.beam_id}_CP{i:03d}.mha' for i in range(self.n_cp)]
        dose = read_dose_parallel(fl, num_threads=8)
        dose = torch.tensor(dose)
        return dose
    

    def rotate_dose(self):
        
        for i in tqdm(range(6), desc='Rotating dose'):
            self.rot_input_tensor[:] = self.dose[30*i:(i+1)*30]
            self.dose[30*i:(i+1)*30] = rotate_image_batched(
                self.rot_input_tensor,  # Shape: (D, H, W) or (1, 1, D, H, W) on GPU
                self.plan.img.GetSpacing(),  # Can now be a torch.Tensor or list/tuple
                self.plan.img.GetOrigin(),  # Can now be a torch.Tensor or list/tuple
                self.plan.isocentre,  # Can now be a torch.Tensor or list/tuple
                -self.angles[i*30:(i+1)*30],  # Can now be a torch.Tensor of angles on GPU
                axis="z",
                bg_value=0,
                pad_voxels=None,
            ).cpu()

    def rotate_img(self):
        img_rot = torch.zeros((180,) + self.img.shape)

        for i in tqdm(range(6), desc='Rotating image'):
            self.rot_input_tensor[:] = self.img.expand(30, -1, -1, -1)
            img_rot[i*30:(i+1)*30] = rotate_image_batched(
                self.rot_input_tensor[:],  # Shape: (D, H, W) or (1, 1, D, H, W) on GPU
                self.plan.img.GetSpacing(),  # Can now be a torch.Tensor or list/tuple
                self.plan.img.GetOrigin(),  # Can now be a torch.Tensor or list/tuple
                self.plan.isocentre,  # Can now be a torch.Tensor or list/tuple
                -self.angles[i*30:(i+1)*30],  # Can now be a torch.Tensor of angles on GPU
                axis="z",
                bg_value=-1024,
                pad_voxels=None,
            ).cpu()

        return img_rot

    def cal_bev_parallel(self, num_threads=8):
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

        return torch.tensor(combined_array) 

    def cal_bev(self, n_batch=30):

        def _cal_scales(self):
            src_mm = get_source_location_mm(self.isocentre, 0, self.sad)
            z_scales = torch.tensor(cal_scales(self.plan.img, self.isocentre, src_mm))[..., None, None]
            return z_scales

        def _cal_mlc_2d(self):
            # Get the 2d bev at isocentre for all 180 control points
            bev_iso_list = []
            for i in range(180):
                mlc = MLC.get_mlc_segs_mm(self.cp_info[i]['l'], self.cp_info[i]['r'], self.isocentre)
                bev_iso = draw_iso_mlc(self.plan.img, mlc) # (nx, nz) -> (246, 249)
                bev_iso_list.append(bev_iso)
            bev_iso_list = np.stack(bev_iso_list, axis=0)
            mlc_masks_2d = torch.tensor(bev_iso_list).to(torch.float32)
            return mlc_masks_2d
        
        # Cal z_scales
        z_scales = _cal_scales(self)
        mlc_masks_2d = _cal_mlc_2d(self)

        isocentre_idx = mm2idx(self.plan.img, [self.isocentre])[0]

        # Set the batch number (# of images to process) and get the grid
        # The grid can be reused for each beam
        grid_5d = make_grid_5d(n_batch, isocentre_idx, z_scales, self.plan.img).cuda()

        print('Grid created')
        
        # Get the beam path by batch
        bevs = []
        for i in tqdm(range(0, 180, n_batch), desc='Calulating BEV beam path'):
            bevs.append(get_bev_torch(mlc_masks_2d[i:(i+n_batch)], grid_5d))
        bevs = torch.cat(bevs, dim=0)

        return bevs
        
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
        






