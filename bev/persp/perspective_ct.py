import numpy as np
import SimpleITK as sitk
from scipy.ndimage import map_coordinates

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
    ed_volume[mask2] = 1.0 + (hu_volume[mask2] * 0.00092) # Typical bone slope
    
    # Clip any extreme negative values safely to absolute vacuum (0.0)
    return np.clip(ed_volume, 0.0, None)


def transform_electron_density(ed_volume, d_source, voxel_spacing):
    """
    Transforms an Electron Density volume from cone-beam to parallel geometry.
    Because background air is 0.0, the Jacobian preserves total value mass perfectly.
    """
    D, H, W = ed_volume.shape
    dx, dy, dz = voxel_spacing
    cx, cy = W / 2.0, H / 2.0
    
    # 1. Coordinate mapping grid setup
    z_indices, y_indices, x_indices = np.mgrid[0:D, 0:H, 0:W]
    x_centered = x_indices - cx
    y_centered = y_indices - cy
    
    # 2. Distance and scaling calculation
    phys_z = d_source + (z_indices * dz)
    t = d_source / phys_z  # Perspective scaling ratio
    
    # 3. Inverse lookup coordinates
    x_lookup = (x_centered * t) + cx
    y_lookup = (y_centered * t) + cy
    z_lookup = z_indices
    
    sampling_coords = np.vstack((
        z_lookup.ravel(),
        y_lookup.ravel(),
        x_lookup.ravel()
    ))
    
    # 4. Trilinear interpolation (order=1)
    # Background air is safely 0.0 outside boundaries
    rectified_ed_flat = map_coordinates(
        ed_volume, 
        sampling_coords, 
        order=1, 
        mode='constant', 
        cval=0.0
    )
    rectified_ed = rectified_ed_flat.reshape(D, H, W)
    
    # 5. Apply Jacobian weight (t squared) to conserve total mass
    return rectified_ed * (t ** 2)


# --- PIPELINE VERIFICATION EXECUTION ---
if __name__ == "__main__":
    # Setup mock CT geometry array (32 slices, 128x128 grid)
    D, H, W = 32, 128, 128
    ct_volume = np.full((D, H, W), -1000.0, dtype=np.float32) # Air background
    
    d_source = 100.0                # Source focal distance in mm
    voxel_spacing = (1.0, 1.0, 1.0) # 1mm isotropic voxels
    cx, cy = W / 2.0, H / 2.0
    
    # Create an expanding soft tissue structure (+50 HU) in the CT scanner grid
    for z in range(D):
        phys_z = d_source + (z * voxel_spacing[2])
        radius = 20.0 * (phys_z / d_source)
        y, x = np.ogrid[:H, :W]
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        ct_volume[z, dist <= radius] = 50.0  # Soft tissue structure
        
    print("--- Step 1: Converting CT (HU) to Electron Density ---")
    ed_volume = hu_to_electron_density(ct_volume)
    print(f"Background value in Electron Density space: {ed_volume[0, 0, 0]:.1f} (Air is now zero)")
    print(f"Initial Total Electron Density Sum:         {np.sum(ed_volume):.2f}")
    
    print("\n--- Step 2: Running Rectification Transform with Jacobian ---")
    parallel_ed_volume = transform_electron_density(ed_volume, d_source, voxel_spacing)
    print(f"Rectified Total Electron Density Sum:       {np.sum(parallel_ed_volume):.2f}")
    
    # Percent difference evaluation check
    diff = abs(np.sum(ed_volume) - np.sum(parallel_ed_volume)) / np.sum(ed_volume) * 100
    print(f"Total conservation error mismatch rate:     {diff:.4f} %")
