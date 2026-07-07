from sympy.geometry import Point3D, Plane
import numpy as np
import SimpleITK as sitk
import json
from tqdm import tqdm
from skimage.draw import polygon
from rotate import rotate_image
import cv2
from scipy.ndimage import label


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


# # skimage version
# def draw_polygon(arr, slice_idx, shape_idx, fill=1):
#     """shape_idx: the 3D indices of the shape"""
#     nz, ny, nx = arr.shape
#     rr, cc = polygon(shape_idx[:, 0], shape_idx[:, 2], shape=(nz, nx))

#     # There could be out of bound idx from polygon
#     # Only keep the in-bound points
#     in_bound = (cc < nz) & (rr < nx)
#     cc, rr = cc[in_bound], rr[in_bound]

#     out = arr.copy()
#     out[cc, slice_idx, rr] = fill

#     return out


def draw_polygon(arr, slice_idx, shape_idx, fill=1):
    """shape_idx: the 3D indices of the shape"""
    out = arr.copy()
    cv2_pts = [
        np.array(idx[:, [0, 2]], dtype=np.int32).reshape((-1, 1, 2))
        for idx in shape_idx
    ]
    cv2_bev = cv2.fillPoly(out[:, slice_idx, :], cv2_pts, color=fill)

    return out


class MLC:
    """A group of functions to calculate the physical coordinates of
    each shape in the MLC
    """

    def get_mlc_2d_offsets_mm(mlc_offsets, leaf_width=5):
        """Convert the raw MLC leaf offsets to physical offsets
        relative to the isocentre.
        Given 80 leafs, and each is 5mm, the span is 400 mm.
        """

        pts_mm = []
        n_leafs = len(mlc_offsets) / 2

        for i, offset in enumerate(mlc_offsets):
            # The number in MLC means moving right
            point_1 = (offset, (i - n_leafs) * leaf_width)
            point_2 = (offset, ((i + 1) - n_leafs) * leaf_width)

            pts_mm.append(point_1)
            pts_mm.append(point_2)

        return np.array(pts_mm)

    def combine_mlc_offsets(mlc_lf, mlc_rt, isocentre, in_3d=False):
        """Combine the absoltue left and right MLC physical offsets
        If in_3d, it will return 3D coords, otherwise 2D coords along BEV
        """

        mlc_lf = mlc_lf.tolist()
        mlc_rt = mlc_rt.tolist()

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

    def get_mlc_segs_mm(mlc_left, mlc_right, isocentre):
        """End-to-end from raw MLC data to each isolated shape from MLC"""

        # Convert to np array for element-wise comparison
        mlc_left, mlc_right = np.array(mlc_left), np.array(mlc_right)

        # Get the is_open mask and repeat it (2 pts per MLC leaf)
        is_open = mlc_left != mlc_right  # 80
        labels_open, n_labels = label(is_open)
        labels = np.repeat(labels_open, 2)  # 160

        mlc_lf = MLC.get_mlc_2d_offsets_mm(mlc_left)  # (160, 2)
        mlc_rt = MLC.get_mlc_2d_offsets_mm(mlc_right)  # (160, 2)

        segs = [
            MLC.combine_mlc_offsets(
                mlc_lf[labels == lab], mlc_rt[labels == lab], isocentre, in_3d=True
            )
            for lab in range(1, n_labels + 1)
        ]

        return segs


# For each CP
class MLCDrawer:
    """BEV, assuming the gantry is rotated to 0 angle"""

    def __init__(self, ref_img, mlc, isocentre, sad):
        self.ref_img = ref_img
        self.mlc = mlc
        self.isocentre = isocentre
        self.isocentre_idx = self.ref_img.TransformPhysicalPointToIndex(self.isocentre)
        self.sad = sad
        self.angle = 0  # BEV (angle rotated to 0)

        self.source_mm = get_source_location_mm(
            self.isocentre, self.angle, self.sad
        )  # source physical location

        self.dist_iso = get_distance_slice_pt(
            self.ref_img, self.isocentre_idx[1], self.source_mm
        )  # source slice to isocentre

    def idx2mm(self, indices):
        return [self.ref_img.TransformIndexToPhysicalPoint(i) for i in indices]

    def mm2idx(self, pts):
        return [self.ref_img.TransformPhysicalPointToIndex(i) for i in pts]

    def cal_ratios(self):
        """Calcalte the ratios relative to distance(iso_slice, source)"""
        nx, ny, nz = self.ref_img.GetSize()
        dist_fist = get_distance_slice_pt(self.ref_img, 0, self.source_mm)
        dist_last = get_distance_slice_pt(self.ref_img, ny - 1, self.source_mm)

        ratios = np.linspace(dist_fist, dist_last, ny) / self.dist_iso

        return ratios

    def cal_bev_beam_path(self):

        img_arr = sitk.GetArrayFromImage(self.ref_img)
        arr = np.zeros(img_arr.shape, np.uint8)

        # Scaling ratios for each slice
        ratios = self.cal_ratios()

        # Use the ratio to draw polygon on each slice
        for i, ratio in enumerate(ratios):

            # Intercepted physical locations for all segments
            intc_shape_pts = [
                get_intercept_points(seg, self.source_mm, ratio) for seg in self.mlc
            ]

            # Intercepted voxel idx for all segments
            intc_shape_idx = [np.array(self.mm2idx(pts)) for pts in intc_shape_pts]

            # Draw the polygon at slice i
            # intc_shape_idx is a collections of shape indices
            arr = draw_polygon(arr, i, intc_shape_idx)

        img = sitk.GetImageFromArray(arr)
        img.CopyInformation(self.ref_img)

        return img


if __name__ == "__main__":
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
    sad = beam0["SAD"]

    ct_rot = rotate_image(
        infile="data/ct.mha",
        isocentre=isocentre,
        degree=-gantry_angle,
    )

    rotate_image(
        infile=f"data/Dose_B0_CP{cp_idx:03d}.mha",
        isocentre=isocentre,
        degree=-gantry_angle,
        outfile=f"data/rotated/d{gantry_angle}.mha",
        bg_value=0,
    )

    # Get the MLC physical positions
    # [shape 1 (nx3), shape 2 (mx3), ...]
    mlc = MLC.get_mlc_segs_mm(mlc_left, mlc_right, isocentre)

    drawer = MLCDrawer(ct_rot, mlc, isocentre, sad)
    bev = drawer.cal_bev_beam_path()

    sitk.WriteImage(ct_rot, f"data/rotated/ct{gantry_angle}.mha")
    sitk.WriteImage(bev, f"data/rotated/bev{gantry_angle}.nii.gz")
