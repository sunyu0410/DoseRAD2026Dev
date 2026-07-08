import numpy as np
import SimpleITK as sitk
from scipy.ndimage import map_coordinates

def mask_cone_to_parallel(cone_mask, d_source, voxel_spacing):
    """
    Transforms an expanding binary cone structure embedded in a standard 
    uniform grid into a straight, parallel cylinder.
    """
    D, H, W = cone_mask.shape
    dx, dy, dz = voxel_spacing
    cx, cy = W / 2.0, H / 2.0
    
    # 1. Build the 3D coordinate meshgrid for the destination parallel volume
    z_indices, y_indices, x_indices = np.mgrid[0:D, 0:H, 0:W]
    
    x_centered = x_indices - cx
    y_centered = y_indices - cy
    
    # 2. Calculate physical depth from focal source point to current voxel slice
    phys_z = d_source + (z_indices * dz)
    
    # 3. Calculate forward scaling factor (magnification)
    magnification = phys_z / d_source
    
    # 4. Pull wide coordinates inward by multiplying by magnification
    x_source_lookup = (x_centered * magnification) + cx
    y_source_lookup = (y_centered * magnification) + cy
    z_source_lookup = z_indices
    
    # Stack layout to shape (3, D*H*W) for index sampling
    sampling_coords = np.vstack((
        z_source_lookup.ravel(),
        y_source_lookup.ravel(),
        x_source_lookup.ravel()
    ))
    
    # 5. Resample the mask using nearest-neighbour (order=0) to keep it binary
    parallel_mask_flat = map_coordinates(
        cone_mask, 
        sampling_coords, 
        order=0, 
        mode='constant', 
        cval=0.0
    )
    
    return parallel_mask_flat.reshape(D, H, W)


def save_as_sitk_nifti(numpy_array, file_path, voxel_spacing):
    """
    Converts a NumPy array into a SimpleITK image, injects physical spacing metadata,
    and writes it directly out to a compressed .nii.gz Nifti file.
    """
    # SimpleITK expects array axis ordering to be (X, Y, Z) / (Width, Height, Depth).
    # NumPy uses (Z, Y, X). Passing it directly automatically handles the axis flip mapping.
    sitk_image = sitk.GetImageFromArray(numpy_array)
    
    # SimpleITK spacing uses explicit (X, Y, Z) ordering
    dx, dy, dz = voxel_spacing
    sitk_image.SetSpacing((float(dx), float(dy), float(dz)))
    
    # Write to disk
    sitk.WriteImage(sitk_image, file_path)
    print(f"Successfully saved NIfTI image: {file_path}")


# --- RUN PIPELINE AND FILE EXPORT ---
if __name__ == "__main__":
    # Define matrix spatial shape configurations
    D, H, W = 32, 128, 128
    uniform_grid_mask = np.zeros((D, H, W), dtype=np.uint8) # Using uint8 for a binary mask file
    
    d_source = 80.0                 # Physical focal distance (mm)
    voxel_spacing = (1.0, 1.0, 1.5) # dx, dy, dz spacings in mm
    cx, cy = W / 2.0, H / 2.0
    
    print("Step 1: Building synthetic expanding cone mask...")
    base_radius = 12.0 
    for z in range(D):
        phys_z = d_source + (z * voxel_spacing[2])
        current_cone_radius = base_radius * (phys_z / d_source)
        
        y, x = np.ogrid[:H, :W]
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        uniform_grid_mask[z, dist <= current_cone_radius] = 1
        
    print("Step 2: Processing perspective transform...")
    rectified_parallel_mask = mask_cone_to_parallel(uniform_grid_mask, d_source, voxel_spacing)
    
    print("\nStep 3: Exporting files via SimpleITK...")
    # Save input expanding cone
    save_as_sitk_nifti(uniform_grid_mask, "cone_input_mask.nii.gz", voxel_spacing)
    
    # Save output rectified parallel cylinder
    save_as_sitk_nifti(rectified_parallel_mask, "parallel_output_mask.nii.gz", voxel_spacing)
