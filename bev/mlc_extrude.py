# From Gemini
# To be studied

import SimpleITK as sitk
import numpy as np
from skimage.draw import polygon

# 1. Setup the geometry of the target isocenter image grid
pixel_spacing = (1.0, 1.0)  # 1mm resolution at isocenter
grid_size = (200, 200)      # 20cm x 20cm field size
isocenter_index = (100, 100) # Index where (0,0) physical coordinate sits

# Create a blank 2D reference image
mask_sitk = sitk.Image(grid_size, sitk.sitkUInt8)
mask_sitk.SetSpacing(pixel_spacing)
# Shift origin so that physical (0,0) maps exactly to our isocenter index
mask_sitk.SetOrigin((-isocenter_index[0]*pixel_spacing[0], -isocenter_index[1]*pixel_spacing[1]))

# 2. Mock MLC Data: relative to Isocenter (in mm)
# Dictionary format: leaf_y_center: (left_leaf_tip_x, right_leaf_tip_x)
leaf_width = 5.0  # 5mm leaves
mlc_data = {
    -10.0: (-20.0, 20.0),
     -5.0: (-30.0, 30.0),
      0.0: (-40.0, 40.0),
      5.0: (-30.0, 30.0),
     10.0: (-20.0, 20.0),
}

# 3. Build the open aperture polygon path from leaf tips
left_bank_points = []
right_bank_points = []

# Sort by Y descending to create a continuous outer path loop
for y_center in sorted(mlc_data.keys(), reverse=True):
    x_left, x_right = mlc_data[y_center]
    y_top = y_center + (leaf_width / 2.0)
    y_bottom = y_center - (leaf_width / 2.0)
    
    # Left bank path goes down
    left_bank_points.append((x_left, y_top))
    left_bank_points.append((x_left, y_bottom))
    
    # Right bank path goes down (will be reversed to complete the loop clock-wise)
    right_bank_points.append((x_right, y_top))
    right_bank_points.append((x_right, y_bottom))

# Combine to form a single closed polygon path loop (Left Bank down -> Right Bank up)
polygon_physical_pts = left_bank_points + right_bank_points[::-1]

# 4. Convert physical coordinates (mm) to voxel array indices
polygon_indices = []
for pt in polygon_physical_pts:
    idx = mask_sitk.TransformPhysicalPointToIndex(pt)
    # skimage requires (row, col) which maps to (Y-index, X-index)
    polygon_indices.append((idx[1], idx[0]))

polygon_indices = np.array(polygon_indices)

# 5. Rasterise the path polygon onto a NumPy array buffer
mask_numpy = np.zeros((grid_size[1], grid_size[0]), dtype=np.uint8)
rr, cc = polygon(polygon_indices[:, 0], polygon_indices[:, 1], shape=mask_numpy.shape)
mask_numpy[rr, cc] = 1  # Burn the filled path aperture as 1

# 6. Convert the filled array back to SimpleITK with matching metadata
extrusion_path_image = sitk.GetImageFromArray(mask_numpy)
extrusion_path_image.CopyInformation(mask_sitk)

# Save or display the MLC aperture path mask
sitk.WriteImage(extrusion_path_image, "mlc_aperture_path.nii.gz")
