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

    def rotate_to_bev(self, angle):
        img_rot = rotate_image(
            infile=self.img_file_path,
            isocentre=self.isocentre,
            degree=angle,
            # outfile=f"data/rotated/ct{gantry_angle}.mha",
        )

        dose_rot = rotate_image(
            infile=str(self.dose_dir / f"Dose_B{self.beam_id}_CP{self.cp_idx:03d}.mha"),
            isocentre=self.isocentre,
            degree=angle,
            # outfile=f"data/rotated/d{gantry_angle}.mha",
            bg_value=0,
        )

        mask_rot = rotate_image_sitk(
            image=self.body_mask,
            isocentre=self.isocentre,
            degree=angle,
            # outfile=f"data/rotated/d{gantry_angle}.mha",
            bg_value=0,
            intpl=sitk.sitkNearestNeighbor,
        )

        return img_rot, dose_rot, mask_rot

    def generate_view(self):
        beam = self.beam_info[self.beam_id]
        cp = self.cp[self.beam_id][self.cp_idx]

        sad = beam["sad"]
        isocentre = beam["isocentre"]
        gantry_angle = cp["ga"]

        img_rot, dose_rot, mask_rot = self.rotate_to_bev(-gantry_angle)

        mlc = MLC.get_mlc_segs_mm(cp["l"], cp["r"], isocentre)

        drawer = MLCDrawer(img_rot, mlc, isocentre, sad)
        bev = drawer.cal_bev_beam_path()

        return img_rot, dose_rot, bev, mask_rot


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
    img_rot, dose_rot, bev, body_mask = plan.generate_view()
    gantry_angle = plan.gantry_angle

    sitk.WriteImage(img_rot, f"data/rotated/ct{gantry_angle}.mha")
    sitk.WriteImage(dose_rot, f"data/rotated/dose{gantry_angle}.mha")
    sitk.WriteImage(bev, f"data/rotated/bev{gantry_angle}.nii.gz")
    sitk.WriteImage(body_mask, f"data/rotated/body_mask{gantry_angle}.nii.gz")

    a = sitk.GetArrayFromImage(img_rot)
    b = sitk.GetArrayFromImage(dose_rot)
    c = sitk.GetArrayFromImage(bev)

    plt.imshow(a[102], alpha=0.4, cmap="gray")
    plt.imshow(b[102], alpha=0.4)
    plt.imshow(b[102], alpha=0.4)
    plt.title(f"CP {cp_idx}")
    plt.savefig(f"data/rotated/png/CP-{cp_idx:3}.png")
    plt.clf()
