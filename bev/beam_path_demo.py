from sympy.geometry import Point3D, Plane
import numpy as np
import SimpleITK as sitk
import json
from tqdm import tqdm
from skimage.draw import polygon
from rotate import rotate_image
from geometry import *


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


# Retrieve info
cp_idx = 23
beam_info = json.load(open("data/1ABB006.json"))
beam0 = beam_info["beams"][0]

assert beam0["control_points"][cp_idx]["cp_idx"] == cp_idx
mlc_left = np.array(beam0["control_points"][cp_idx]["mlc_left_int_mm"])
mlc_right = np.array(beam0["control_points"][cp_idx]["mlc_right_int_mm"])
isocentre = np.array(beam0["iso_center"])
gantry_angle = beam0["control_points"][cp_idx]["gantry_angle"]


rotate_image(
    infile="data/ct.mha",
    isocentre=isocentre,
    degree=gantry_angle,
    outfile=f"data/rotated/ct{gantry_angle}.mha",
)


rotate_image(
    infile=f"data/Dose_B0_CP{cp_idx:03d}.mha",
    isocentre=isocentre,
    degree=gantry_angle,
    outfile=f"data/rotated/d{gantry_angle}.mha",
    bg_value=0,
)

# After rotation
# Reset gantry angle to 0
angle = 0

# Get the MLC physical positions
mlc_lf = get_mlc_offsets_mm(mlc_left)
mlc_rt = get_mlc_offsets_mm(mlc_right)

mlc = get_mlc_mm(mlc_lf, mlc_rt, isocentre, in_3d=True)

# ct = sitk.ReadImage("data/ct_1mm.mha")
ct = sitk.ReadImage(f"data/rotated/ct{gantry_angle}.mha")
nx, ny, nz = ct.GetSize()
ct_arr = sitk.GetArrayFromImage(ct)
arr = np.zeros(ct_arr.shape, np.uint8)

isocentre_idx = ct.TransformPhysicalPointToIndex(isocentre)
source_mm = get_source_location_mm(isocentre, angle)
dist_iso = get_distance_slice_pt(ct, isocentre_idx[1], source_mm)

for i in tqdm(range(ny)):
    shape_idx = (0, i, 0), (0, i, nz - 1), (nx - 1, i, 0)
    shape_pts = [ct.TransformIndexToPhysicalPoint(i) for i in shape_idx]
    dist = get_distance_plane_pt(shape_pts, source_mm)
    ratio = dist / dist_iso

    intc_shape_pts = get_intercept_points(mlc, source_mm, ratio)
    intc_shape_idx = [ct.TransformPhysicalPointToIndex(i) for i in intc_shape_pts]
    intc_shape_idx = np.array(intc_shape_idx)

    rr, cc = polygon(intc_shape_idx[:, 0], intc_shape_idx[:, 2], shape=(nz, nx))

    # There could be out of bound idx from polygon
    # Only keep the in-bound points
    in_bound = (cc < nz) & (rr < nx)
    cc, rr = cc[in_bound], rr[in_bound]

    arr[cc, i, rr] = 1

img = sitk.GetImageFromArray(arr)
img.CopyInformation(ct)
sitk.WriteImage(img, f"data/rotated/intersection{gantry_angle}.nii.gz")
