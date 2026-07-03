from sympy.geometry import Point3D, Plane
import numpy as np
import SimpleITK as sitk
import json
from tqdm import tqdm
from skimage.draw import polygon
from rotate import rotate_image
from geometry import *


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

drawer = MLCDrawer(ct, mlc, isocentre)

bev = drawer.cal_bev_beam_path()
sitk.WriteImage(bev, f"data/rotated/bev{gantry_angle}.nii.gz")
