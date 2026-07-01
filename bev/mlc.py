import json
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from skimage.draw import polygon

beam_info = json.load(open('data/1ABB006.json'))

beam0 = beam_info['beams'][0]

mlc_left = np.array(beam0['control_points'][0]['mlc_left_int_mm'])
mlc_right = np.array(beam0['control_points'][0]['mlc_right_int_mm'])
isocentre = np.array(beam0['iso_center'])

def get_mlc_offsets(mlc_offsets, leaf_width=5):

    pts_mm = []
    n_leafs = len(mlc_offsets) / 2


    for i, offset in enumerate(mlc_offsets):

        point_1 = (offset, (n_leafs-i)*leaf_width)
        point_2 = (offset, (n_leafs-(i+1))*leaf_width)

        pts_mm.append(point_1)
        pts_mm.append(point_2)

    # pts_mm = np.array(pts_mm)

    return pts_mm

mlc_lf = get_mlc_offsets(mlc_left)
mlc_rt = get_mlc_offsets(mlc_right)

mlc = mlc_lf + mlc_rt[::-1]
mlc = np.array(mlc)

print(mlc)

plt.plot(mlc[:,0], mlc[:,1])
plt.axis('equal')
plt.show()

ct = sitk.ReadImage('data/ct_1mm.mha')
mlc_idx = np.array([ct.TransformPhysicalPointToIndex((*i, isocentre[2])) for i in mlc])

# (492, 492, 498)
mlc_voxels = np.zeros(ct.GetSize()[::-1], dtype=np.uint8)
rr, cc = polygon(mlc_idx[:, 0], mlc_idx[:, 1], shape=(498, 492))
mlc_voxels[cc, :, rr] = 1  # Burn the filled path aperture as 1

mlc_sitk = sitk.GetImageFromArray(mlc_voxels)
mlc_sitk.CopyInformation(ct)

sitk.WriteImage(mlc_sitk, 'data/mlc_extrusion.mha')

# #######

# leaf_width = 5 # mm

# left_pts_mm = []
# n_leafs = len()
# for i, left_offset in enumerate(mlc_left_int_mm):
#     point_1 = (-left_offset, (40-i)*leaf_width)
#     point_2 = (-left_offset, (40-(i+1))*leaf_width)

#     left_pts_mm.append(point_1)
#     left_pts_mm.append(point_2)

# left_pts_mm = np.array(left_pts_mm)

# left_pts_mm += isocentre # array([-46.84718448,  27.77766326])