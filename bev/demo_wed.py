# This is the correct demo

def hu_to_electron_density(hu_volume):
    """
    Converts Hounsfield Units (HU) to Relative Electron Density (RED).
    Air (-1000 HU) becomes 0.0. Water (0 HU) becomes 1.0.
    """
    ed_volume = np.zeros_like(hu_volume, dtype=np.float32)

    # Standard clinical dual-linear calibration curve mapping
    # Curve part 1: Air to Water (-1000 to 0 HU)
    mask1 = hu_volume <= 0
    ed_volume[mask1] = 1.0 + (hu_volume[mask1] / 1000.0)

    # Curve part 2: Water to Dense Bone (0 to 2000 HU)
    mask2 = hu_volume > 0
    ed_volume[mask2] = 1.0 + (hu_volume[mask2] * 0.00092)  # Typical bone slope

    # Clip any extreme negative values safely to absolute vacuum (0.0)
    return np.clip(ed_volume, 0.0, None)


def perspective_transform(arr, scale_centre, scale_factor):
    Z, Y, X = arr.shape
    x, y, z = scale_centre

    z_indices, y_indices, x_indices = np.mgrid[0:Z, 0:Y, 0:X]

    z_lookup = (z_indices - z) * t + z
    y_lookup = y_indices
    x_lookup = (x_indices - x) * t + x

    sampling_coords = np.vstack((z_lookup.ravel(), y_lookup.ravel(), x_lookup.ravel()))

    out = map_coordinates(
        arr, sampling_coords, order=1, mode="constant", cval=0.0
    ).reshape(Z, Y, X) * (t ** 2)

    return out

# Get the cropped data and calculated electron density
# TODO: consider the case where the isocentre XZ projection is outside the cropped region
(zmin, ymin, xmin), (zmax, ymax, xmax) = plan.get_bbox(
    sitk.GetArrayFromImage(plan.bev), margin=5
)

arr_bev = sitk.GetArrayFromImage(
    plan.bev[xmin : xmax + 1, ymin : ymax + 1, zmin : zmax + 1]
)

arr_ct = sitk.GetArrayFromImage(
    plan.img[xmin : xmax + 1, ymin : ymax + 1, zmin : zmax + 1]
)

arr_ed = hu_to_electron_density(arr_ct)
arr_ed *= arr_bev

# Perspective Transform
# Scaling centre and factor
dz, dy, dx = plan.bev.GetSpacing()
z, y, x = np.array(plan.isocentre_idx[::-1]) - np.array([zmin, ymin, xmin])

distance_to_first = plan.sad - y * dy
t = (distance_to_first + y_indices * dy) / plan.sad  # so that at 63 it's 1000

rectified_ed = perspective_transform(arr_ed, (x, y, z), t)

# print(rectified_ed.sum() == 139027.72115043824)
