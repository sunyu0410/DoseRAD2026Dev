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
        # The number in MLC means moving right
        # But the mha is LPS
        # Hence it's -1*offset 
        point_1 = (-offset, (i-n_leafs)*leaf_width)
        point_2 = (-offset, ((i+1)-n_leafs)*leaf_width)

        pts_mm.append(point_1)
        pts_mm.append(point_2)

    # pts_mm = np.array(pts_mm)

    return pts_mm
    

mlc_lf = get_mlc_offsets(mlc_left)
mlc_rt = get_mlc_offsets(mlc_right)

mlc = mlc_lf + mlc_rt[::-1]
mlc = np.array(mlc)
mlc[:,0] += isocentre[0]
mlc[:,1] += isocentre[2]


print(mlc)

plt.plot(mlc[:,0], mlc[:,1])
plt.axis('equal')
plt.show()

ct = sitk.ReadImage('data/ct_1mm.mha')
mlc_idx = np.array([ct.TransformPhysicalPointToIndex((i[0], isocentre[1], i[1])) for i in mlc])

nx, ny, nz = ct.GetSize() # (498, 492, 492)

rr, cc = polygon(mlc_idx[:, 0], mlc_idx[:, 2], shape=(nz, nx))
mlc_voxels = np.zeros((nz, ny, nx), dtype=np.uint8) # (492, 492, 498)
mlc_voxels[cc, :, rr] = 1  # Burn the filled path aperture as 1

mlc_sitk = sitk.GetImageFromArray(mlc_voxels)
mlc_sitk.CopyInformation(ct)

sitk.WriteImage(mlc_sitk, 'data/mlc_extrusion.mha')

