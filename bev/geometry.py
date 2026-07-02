from sympy.geometry import Point3D, Plane
import numpy as np
import SimpleITK as sitk
import json
from mlc import get_mlc_offsets_mm, get_mlc_mm
from tqdm import tqdm
from skimage.draw import polygon


def get_intercept_points(shape_pts, source_pt, ratio):
    """The ratio is the Distance(plane, source) / Distance(reference shape, source)"""
    return source_pt + ratio * (shape_pts - source_pt)


def get_distance_plane_pt(plane_pts, target_point):
    plane = Plane(*[Point3D(i) for i in plane_pts])
    target_point = Point3D(target_point)
    distance = float(plane.distance(target_point))

    return distance


def get_source_location_mm(isocentre, gantry_angle_deg, sad_mm=1000.0):
    """
    Computes the absolute 3D physical coordinate [X, Y, Z] of the radiation
    source in LPS space for a gantry rotating around the longitudinal Z-axis.

    Parameters:
    isocentre (list or np.array): The physical [X, Y, Z] coordinates
                                      of the isocentre in millimetres.
    gantry_angle_deg (float):         The gantry angle directly from the JSON.
    sad_mm (float):                   Source-to-Axis Distance (Default: 1000.0 mm).
    """
    angle_rad = np.radians(-gantry_angle_deg)

    dx = -sad_mm * np.sin(angle_rad)
    dy = -sad_mm * np.cos(angle_rad)
    dz = 0.0

    offset_vector = np.array([dx, dy, dz])

    source_lps = np.array(isocentre) + offset_vector

    return source_lps


def get_distance_slice_pt(img, slice_idx, pt):
    nx, ny, nz = img.GetSize()
    shape_idx = (0, slice_idx, 0), (0, slice_idx, nz - 1), (nx - 1, slice_idx, 0)
    shape_pts = [img.TransformIndexToPhysicalPoint(i) for i in shape_idx]
    dist = get_distance_plane_pt(shape_pts, pt)
    return dist


isocentre = np.array([-46.84718448, 27.77766326, -28.13538836])
print(get_source_location_mm(isocentre, -180))
print(get_source_location_mm(isocentre, -150))
print(get_source_location_mm(isocentre, 0))

# Retrieve info
cp_idx = 23
beam_info = json.load(open("data/1ABB006.json"))
beam0 = beam_info["beams"][0]

assert beam0["control_points"][cp_idx]["cp_idx"] == cp_idx
mlc_left = np.array(beam0["control_points"][cp_idx]["mlc_left_int_mm"])
mlc_right = np.array(beam0["control_points"][cp_idx]["mlc_right_int_mm"])
isocentre = np.array(beam0["iso_center"])
gantry_angle = beam0["control_points"][cp_idx]["gantry_angle"]

# Get the MLC physical positions
mlc_lf = get_mlc_offsets_mm(mlc_left)
mlc_rt = get_mlc_offsets_mm(mlc_right)

mlc = get_mlc_mm(mlc_lf, mlc_rt, isocentre, in_3d=True)

ct = sitk.ReadImage("data/ct_1mm.mha")
nx, ny, nz = ct.GetSize()
ct_arr = sitk.GetArrayFromImage(ct)
arr = np.zeros(ct_arr.shape, np.uint8)

isocentre_idx = ct.TransformPhysicalPointToIndex(isocentre)
source_mm = get_source_location_mm(isocentre, gantry_angle)
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
    in_bound = (cc<nz) & (rr<nx)
    cc, rr = cc[in_bound], rr[in_bound]
    
    arr[cc, i, rr] = 1

img = sitk.GetImageFromArray(arr)
img.CopyInformation(ct)
sitk.WriteImage(img, "intercepted.nii.gz")
