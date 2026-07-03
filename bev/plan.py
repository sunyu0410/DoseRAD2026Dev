import numpy as np
import SimpleITK as sitk
from pathlib import Path
import json
from collections import OrderedDict as odict
from rotate import rotate_image
from mlc import *
from geometry import *


class Plan:
    def __init__(self, img_file_path, info_json_path, dose_dir):
        self.img_file_path = img_file_path
        self.info_json_path = info_json_path
        self.dose_dir = dose_dir

        self.img = sitk.ReadImage(img_file_path)
        self.info = json.load(open(info_json_path))

        self.parse_json()

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
                    [
                        (
                            cp["cp_idx"],
                            dict(
                                ga=cp["gantry_angle"],
                                l=cp["mlc_left_int_mm"],
                                r=cp["mlc_right_int_mm"],
                            ),
                        )
                        for cp in self.info["beams"][beam_id]["control_points"]
                    ]
                ),
            )
            for beam_id in range(len(self.beam_info))
        )

    def generate_view(self, beam_idx, cp_idx):
        beam = self.beam_info[beam_idx]
        cp = self.cp[beam_idx][cp_idx]

        isocentre = beam["isocentre"]
        sad = beam["sad"]
        gantry_angle = cp["ga"]

        rotate_image(
            infile=self.img_file_path,
            isocentre=isocentre,
            degree=gantry_angle,
            outfile=f"data/class/ct_b{beam_idx}_cp{cp_idx}.nii.gz",
        )

        rotate_image(
            infile=f"data/Dose_B0_CP{cp_idx:03d}.mha",
            isocentre=isocentre,
            degree=gantry_angle,
            outfile=f"data/class/dose_b{beam_idx}_cp{cp_idx}.nii.gz",
            bg_value=0,
        )

        angle = 0

        mlc_lf = get_mlc_offsets_mm(cp["l"])
        mlc_rt = get_mlc_offsets_mm(cp["r"])

        mlc = get_mlc_mm(mlc_lf, mlc_rt, isocentre, in_3d=True)

        nx, ny, nz = self.img.GetSize()
        img_arr = sitk.GetArrayFromImage(self.img)
        arr = np.zeros(img_arr.shape, np.uint8)

        isocentre_idx = self.img.TransformPhysicalPointToIndex(isocentre)
        source_mm = get_source_location_mm(isocentre, angle)
        dist_iso = get_distance_slice_pt(self.img, isocentre_idx[1], source_mm)

        for i in tqdm(range(ny)):
            shape_idx = (0, i, 0), (0, i, nz - 1), (nx - 1, i, 0)
            shape_pts = [self.img.TransformIndexToPhysicalPoint(i) for i in shape_idx]
            dist = get_distance_plane_pt(shape_pts, source_mm)
            ratio = dist / dist_iso

            intc_shape_pts = get_intercept_points(mlc, source_mm, ratio)
            intc_shape_idx = [
                self.img.TransformPhysicalPointToIndex(i) for i in intc_shape_pts
            ]
            intc_shape_idx = np.array(intc_shape_idx)

            rr, cc = polygon(intc_shape_idx[:, 0], intc_shape_idx[:, 2], shape=(nz, nx))

            # There could be out of bound idx from polygon
            # Only keep the in-bound points
            in_bound = (cc < nz) & (rr < nx)
            cc, rr = cc[in_bound], rr[in_bound]

            arr[cc, i, rr] = 1

        img = sitk.GetImageFromArray(arr)
        img.CopyInformation(self.img)
        sitk.WriteImage(img, f"data/class/bev_b{beam_idx}_cp{cp_idx}.nii.gz")


if __name__ == "__main__":
    plan = Plan(
        img_file_path=r"data\photon\training\1ABB006\image\ct.mha",
        info_json_path=r"data\photon\training\1ABB006\1ABB006.json",
        dose_dir=r"data\photon\training\1ABB006\dose",
    )
    print(plan.beam_info)
    plan.generate_view(0, 23)
