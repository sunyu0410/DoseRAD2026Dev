import numpy as np
import SimpleITK as sitk
from pathlib import Path
import json
from collections import OrderedDict as odict
from utils.rotate import rotate_image, rotate_image_sitk
from utils.rotate_torch import rotate_image_batched
from geometry import *
import matplotlib.pyplot as plt
from ct import get_body_mask
import torch


class Plan:
    def __init__(self, img_file_path, info_json_path, dose_dir):
        self.img_file_path = img_file_path          # CT path
        self.img = sitk.ReadImage(img_file_path)    # CT sitk
        self.img_arr = self.file2arr(img_file_path) # CT numpy
        
        self.info_json_path = info_json_path 
        self.info = json.load(open(info_json_path)) # -> parsed into self.beam_info, cp, n_beams

        self.dose_dir = Path(dose_dir)

        # self.body_mask = get_body_mask(self.img, thres=-1024)

        self.parse_json()

        self.create_data_chunks()
        self.cache = dict()

    @staticmethod
    def and_filter(x, y):
        return sitk.AndImageFilter().Execute(x, y)

    def create_data_chunks(self):
        self.groups = []
        for i in range(self.n_beams):
            group = [(i, j) for j in torch.split(torch.arange(self.beam_info[i]["n_cp"]), 10)]
            self.groups.extend(group)

    def get(self, beam_id, cp_idx):
        # Get which group it belongs to
        match = [g for g in self.groups if g[0]==beam_id and cp_idx in g[1]]
        assert len(match) == 1

        # Get the key
        cp_list = match[0][1]
        key = (beam_id, cp_list)
        
        # Get the relative position of the sample
        idx = torch.where(cp_list==cp_idx)[0].item()

        # If not there, fetch it
        if key not in self.cache:
            print('Fetching data for beam_id:', beam_id, 'cp_list:', cp_list)
            data = self.get_batch(beam_id, cp_list.tolist())
            self.cache[key] = data

        # Return it
        img_rot, dose_rot, mask_rot, bev = self.cache[key]
        return img_rot[idx], dose_rot[idx], mask_rot[idx], bev[idx]
            

    @staticmethod
    def file2arr(filepath):
        img = sitk.ReadImage(filepath)
        arr = sitk.GetArrayFromImage(img)
        return arr


    def parse_json(self):
        # In [113]: beam_info[0]
        # Out[113]:
        # {'SAD': 1000,
        # 'isocentre': [-46.8471844842125, 27.777663262437926, -28.13538836315937],
        # 'n_mlc_leaf': 80}
        self.beam_info = odict(
            (
                b["beam_idx"],
                dict(
                    sad=b["SAD"],
                    isocentre=b["iso_center"],
                    n_mlc_leaf=b["num_mlc_leaf_pairs"],
                    n_cp=len(b["control_points"]),
                ),
            )
            for b in self.info["beams"]
        )
        self.cp = odict(
            (
                beam_id,
                odict(
                    (
                        cp["cp_idx"],
                        dict(
                            ga=cp["gantry_angle"],
                            l=cp["mlc_left_int_mm"],
                            r=cp["mlc_right_int_mm"],
                        ),
                    )
                    for cp in self.info["beams"][beam_id]["control_points"]
                ),
            )
            for beam_id in range(len(self.beam_info))
        )
        self.n_beams = len(self.beam_info)

    @staticmethod
    def get_bbox(arr, margin=5):
        """Get the bounding box of an np mask
        results in np array as (zmin, ymin, xmin), (zmax, ymax, xmax)
        """

        shape = arr.shape
        idx = np.array(np.where(arr > 0))

        # Safety check
        min_idx = np.clip(idx.min(1) - margin, 0, shape)
        max_idx = np.clip(idx.max(1) + margin, 0, shape)

        return np.stack([min_idx, max_idx])

    def get_batch(self, beam_id, cp_list):
        # beam_id, cp_list, self = 0, range(10), plan

        beam = self.beam_info[beam_id]
        sad = beam["sad"]
        isocentre = self.beam_info[beam_id]["isocentre"]

        # CT
        degrees_list = torch.tensor([self.cp[beam_id][cp_idx]['ga'] for cp_idx in cp_list])
        img_rot = rotate_image_batched(
            tensor_img = torch.tensor(self.img_arr).cuda().to(torch.float16),
            spacing = self.img.GetSpacing(),
            origin = self.img.GetOrigin(),
            isocentre = isocentre,
            degrees_list = degrees_list,
            axis='z',
            bg_value=-1024,
            pad_voxels=None
        )
        img_rot = img_rot.cpu()
        torch.cuda.empty_cache()

        # Dose
        dose_arr = np.stack([self.file2arr(str(self.dose_dir / f"Dose_B{beam_id}_CP{cp_idx:03d}.mha")) for cp_idx in cp_list], axis=0)
        dose_rot = rotate_image_batched(
            tensor_img = torch.tensor(dose_arr).unsqueeze(1).cuda().to(torch.float16),
            spacing = self.img.GetSpacing(),
            origin = self.img.GetOrigin(),
            isocentre = isocentre,
            degrees_list = degrees_list,
            axis='z',
            bg_value=0,
            pad_voxels=None
        )
        dose_rot = dose_rot.cpu()
        torch.cuda.empty_cache()

        # Mask
        mask_rot = img_rot > -1024

        # BEV
        bev = []
        for i,cp_idx in enumerate(cp_list):
            cp = self.cp[beam_id][cp_idx]
            mlc = MLC.get_mlc_segs_mm(cp["l"], cp["r"], isocentre)
            drawer = MLCDrawer(self.img, mlc, isocentre, sad)
            bev.append(drawer.cal_bev_beam_path())
        bev = torch.tensor(np.stack(bev, axis=0))

        return img_rot, dose_rot, mask_rot, bev



if __name__ == "__main__":
    # plan = Plan(
    #     img_file_path=r"data/DoseRAD2026/photon/training/1ABB006/image/ct.mha",
    #     info_json_path=r"data/DoseRAD2026/photon/training/1ABB006/1ABB006.json",
    #     dose_dir=r"data/DoseRAD2026/photon/training/1ABB006/dose",
    # )

    plan = Plan(
        img_file_path=r"data/ct.mha",
        info_json_path=r"data/1ABB006.json",
        dose_dir=r"data",
    )
    print(plan.beam_info)


    img_rot, dose_rot, mask_rot, bev = plan.get(beam_id=0, cp_idx=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_rot[102, :, :], alpha=0.4, cmap="gray")
    # plt.imshow(dose_rot[102, :, :], alpha=0.4)
    # plt.imshow(mask_rot[102, :, :], alpha=0.4)
    # plt.imshow(bev[102, :, :], alpha=0.4)
    # plt.savefig(f"preview.png")
    # plt.clf()

    ref_img = sitk.ReadImage('data/ct.mha')
    img_rot_sitk =  sitk.GetImageFromArray(img_rot.numpy().astype(np.float32))
    dose_rot_sitk =  sitk.GetImageFromArray(dose_rot.numpy().astype(np.float32))
    mask_rot_sitk =  sitk.GetImageFromArray(mask_rot.numpy().astype(np.uint8))
    bev_sitk =  sitk.GetImageFromArray(bev.numpy().astype(np.uint8))

    img_rot_sitk.CopyInformation(ref_img)
    dose_rot_sitk.CopyInformation(ref_img)
    mask_rot_sitk.CopyInformation(ref_img)
    bev_sitk.CopyInformation(ref_img)

    sitk.WriteImage(img_rot_sitk, f"data/rotated/torch/ct_-180.nii.gz")
    sitk.WriteImage(dose_rot_sitk, f"data/rotated/torch/dose_-180.nii.gz")
    sitk.WriteImage(mask_rot_sitk, f"data/rotated/torch/body_mask_-180.nii.gz")
    sitk.WriteImage(bev_sitk, f"data/rotated/torch/bev_-180.nii.gz")

    # torch.save(plan.tensors, f"data/rotated/tensors.pt")
