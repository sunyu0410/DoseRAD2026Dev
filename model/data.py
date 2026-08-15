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


class PlanData(Dataset):
    def __init__(self, plan):
        self.plan = plan
        self.n_beam = plan.n_beams
        self.ids = [(i, j) for i in range(self.plan.n_beams) for j in range(self.plan.beam_info[i]['n_cp'])]
        self.rot_input_tensor = torch.randn((30,)+self.plan.img.GetSize()[::-1]).to(torch.float16).cuda()
        self.load_doses()
        self.rotate_dose()

    def load_doses(self):
        self.doses = []
        for beam_idx in range(self.plan.n_beams):
            fl = [f'{self.plan.dose_dir}/Dose_B{beam_idx}_CP{i:03d}.mha' for i in range(self.plan.beam_info[beam_idx]['n_cp'])]
            dose = read_dose_parallel(fl, num_threads=32)
            dose = torch.tensor(dose)

            # dose = torch.randn((180, 246, 246, 249)) # Simulate

            self.doses.append(dose)
            break
    

    def rotate_dose(self):
        self.doses_rot = []
        for beam_idx, dose in enumerate(self.doses):
            angles = torch.tensor([-c['ga'] for c in self.plan.cp[beam_idx].values()]).cuda()

            for i in range(6):
                self.rot_input_tensor[:] = dose[30*i:(i+1)*30]
                d_rot = rotate_image_batched(
                    self.rot_input_tensor,  # Shape: (D, H, W) or (1, 1, D, H, W) on GPU
                    self.plan.img.GetSpacing(),  # Can now be a torch.Tensor or list/tuple
                    self.plan.img.GetOrigin(),  # Can now be a torch.Tensor or list/tuple
                    self.plan.isocentre,  # Can now be a torch.Tensor or list/tuple
                    angles[i*30:(i+1)*30],  # Can now be a torch.Tensor of angles on GPU
                    axis="z",
                    bg_value=0,
                    pad_voxels=None,
                ).cpu()

                self.doses[beam_idx][30*i:(i+1)*30] = d_rot
                print('Beam', beam_idx, 'rotated:', 30*i, (i+1)*30)
                
        
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        beam_id, cp_idx = self.ids[idx]
        
        return self.doses[beam_id][cp_idx]

if __name__ == '__main__':
    plan = Plan(
        img_file_path=r"/workspace/DoseRAD2026Dev/data/DoseRAD2026/photon/training/1ABB006/image/ct.mha",
        info_json_path=r"/workspace/DoseRAD2026Dev/data/DoseRAD2026/photon/training/1ABB006/1ABB006.json",
        dose_dir=r"/workspace/DoseRAD2026Dev/data/DoseRAD2026/photon/training/1ABB006/dose",
    )
    print(plan.beam_info)

    dataset = PlanData(plan)
        






