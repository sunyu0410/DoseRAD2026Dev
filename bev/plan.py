import numpy as np
import SimpleITK as sitk
from pathlib import Path
import json
from collections import OrderedDict as odict
from rotate import rotate_image, rotate_image_sitk
from geometry import *
import matplotlib.pyplot as plt
from ct import get_body_mask


class Plan:
    def __init__(self, img_file_path, info_json_path, dose_dir):
        self.img_file_path = img_file_path
        self.info_json_path = info_json_path
        self.dose_dir = Path(dose_dir)

        self.img = sitk.ReadImage(img_file_path)
        self.body_mask = get_body_mask(self.img, thres=-1024)
        self.info = json.load(open(info_json_path))

        self.and_filter = lambda x, y: sitk.AndImageFilter().Execute(x, y)

        self.parse_json()

        self.set_state(beam_id=0, cp_idx=0)

    @property
    def isocentre(self):
        return self.beam_info[self.beam_id]["isocentre"]

    @property
    def gantry_angle(self):
        return self.cp[self.beam_id][self.cp_idx]["ga"]

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

    def set_state(self, beam_id, cp_idx):
        assert cp_idx in self.cp[beam_id]
        self.beam_id = beam_id
        self.cp_idx = cp_idx

        self.rotate_to_bev(masked_to_body=True)

    def rotate_to_bev(self, masked_to_body=True):

        beam = self.beam_info[self.beam_id]
        cp = self.cp[self.beam_id][self.cp_idx]

        self.sad = beam["sad"]
        isocentre = self.isocentre
        gantry_angle = self.gantry_angle

        # Rotate images and store them as self attributes
        angle = -self.gantry_angle  # Reverse the gantry angle

        self.img_rot = rotate_image(
            infile=self.img_file_path,
            isocentre=isocentre,
            degree=angle,
        )

        self.dose_rot = rotate_image(
            infile=str(self.dose_dir / f"Dose_B{self.beam_id}_CP{self.cp_idx:03d}.mha"),
            isocentre=isocentre,
            degree=angle,
            bg_value=0,
        )

        # Rotate an existing sitk image, not from file, hence using rotate_image_sitk
        self.mask_rot = rotate_image_sitk(
            image=self.body_mask,
            isocentre=isocentre,
            degree=angle,
            bg_value=0,
            intpl=sitk.sitkNearestNeighbor,
        )

        # Get the BEV
        self.mlc = MLC.get_mlc_segs_mm(cp["l"], cp["r"], self.isocentre)
        drawer = MLCDrawer(self.img_rot, self.mlc, self.isocentre, self.sad)
        self.bev = drawer.cal_bev_beam_path()
        if masked_to_body:
            self.bev = self.and_filter(self.bev, self.mask_rot)

        # Can then access self.img_rot, self.dose_rot, self.mask_rot and self.bev


if __name__ == "__main__":
    plan = Plan(
        img_file_path=r"data/ct.mha",
        info_json_path=r"data/1ABB006.json",
        dose_dir=r"data",
    )
    print(plan.beam_info)

    # In [5]: img_rot.TransformPhysicalPointToIndex(plan.isocentre)
    # Out[5]: (102, 139, 109)

    beam_id, cp_idx = 0, 23
    plan.set_state(beam_id=beam_id, cp_idx=cp_idx)

    ga = plan.gantry_angle

    sitk.WriteImage(plan.img_rot, f"data/rotated/ct{ga}.mha")
    sitk.WriteImage(plan.dose_rot, f"data/rotated/dose{ga}.mha")
    sitk.WriteImage(plan.mask_rot, f"data/rotated/body_mask{ga}.nii.gz")
    sitk.WriteImage(plan.bev, f"data/rotated/bev{ga}.nii.gz")

    a = sitk.GetArrayFromImage(plan.img_rot)
    b = sitk.GetArrayFromImage(plan.dose_rot)
    c = sitk.GetArrayFromImage(plan.mask_rot)
    d = sitk.GetArrayFromImage(plan.bev)

    plt.imshow(a[:, 139, :], alpha=0.4, cmap="gray")
    plt.imshow(b[:, 139, :], alpha=0.4)
    plt.imshow(c[:, 139, :], alpha=0.4)
    plt.imshow(d[:, 139, :], alpha=0.4)
    plt.title(f"CP {cp_idx}")
    plt.savefig(f"data/rotated/png/CP-{cp_idx:3}.png")
    plt.clf()
