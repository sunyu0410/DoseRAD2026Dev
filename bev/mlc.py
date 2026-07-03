import json
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from skimage.draw import polygon


def get_mlc_offsets_mm(mlc_offsets, leaf_width=5):

    pts_mm = []
    n_leafs = len(mlc_offsets) / 2

    for i, offset in enumerate(mlc_offsets):
        # The number in MLC means moving right
        point_1 = (offset, (i - n_leafs) * leaf_width)
        point_2 = (offset, ((i + 1) - n_leafs) * leaf_width)

        pts_mm.append(point_1)
        pts_mm.append(point_2)

    return pts_mm


def get_mlc_mm(mlc_lf, mlc_rt, isocentre, in_3d=False):

    mlc = mlc_lf + mlc_rt[::-1]
    mlc = np.array(mlc)
    if in_3d:
        # Add second col as 0
        mlc = np.insert(mlc, 1, values=0, axis=1)
    else:
        # Remove the second index
        isocentre = (isocentre[0], isocentre[2])

    mlc += isocentre

    return mlc


def get_mlc_2d_vox(ref_file_path, mlc, fill=1):
    ref = sitk.ReadImage(ref_file_path)
    mlc_idx = np.array(
        [ref.TransformPhysicalPointToIndex((i[0], isocentre[1], i[1])) for i in mlc]
    )

    nx, ny, nz = ref.GetSize()

    rr, cc = polygon(mlc_idx[:, 0], mlc_idx[:, 2], shape=(nz, nx))
    mlc_voxels = np.zeros((nz, nx), dtype=np.uint8)
    mlc_voxels[cc, rr] = fill  # Burn the filled path aperture as 1
    return mlc_voxels


if __name__ == "__main__":

    # Retrieve info
    beam_info = json.load(open("data/1ABB006.json"))
    beam0 = beam_info["beams"][0]

    mlc_left = np.array(beam0["control_points"][0]["mlc_left_int_mm"])
    mlc_right = np.array(beam0["control_points"][0]["mlc_right_int_mm"])
    isocentre = np.array(beam0["iso_center"])

    # Get the MLC physical positions
    mlc_lf = get_mlc_offsets_mm(mlc_left)
    mlc_rt = get_mlc_offsets_mm(mlc_right)

    mlc = get_mlc_mm(mlc_lf, mlc_rt, isocentre)
