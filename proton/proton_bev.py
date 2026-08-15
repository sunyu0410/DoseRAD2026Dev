import torch
import torch.nn.functional as F
import SimpleITK as sitk
import json
import SimpleITK as sitk

def rotate_image_batched(
    tensor_img,  # Shape: (D, H, W) or (1, 1, D, H, W) on GPU
    spacing,  # Can now be a torch.Tensor or list/tuple
    origin,  # Can now be a torch.Tensor or list/tuple
    isocentre,  # Can now be a torch.Tensor or list/tuple
    degrees_list,  # Can now be a torch.Tensor of angles on GPU
    axis="z",
    bg_value=-1024,
    pad_voxels=None,
):
    """
    100% pure PyTorch GPU implementation. No NumPy operations.
    Processes all control points concurrently on the GPU.
    """
    dtype = tensor_img.dtype

    # 1. Standardise tensor dimensions to 5D format: (1, 1, D, H, W)
    if tensor_img.dim() == 3:
        tensor_img = tensor_img.unsqueeze(0).unsqueeze(0)
    elif tensor_img.dim() == 4:
        tensor_img = tensor_img.unsqueeze(1)

    _, _, D, H, W = tensor_img.shape
    device = tensor_img.device

    # 2. Convert metadata inputs directly to PyTorch tensors on the same GPU
    size_xyz = torch.tensor([W, H, D], dtype=dtype, device=device)
    spacing_xyz = torch.as_tensor(spacing, dtype=dtype, device=device)
    origin_xyz = torch.as_tensor(origin, dtype=dtype, device=device)
    iso_xyz = torch.as_tensor(isocentre, dtype=dtype, device=device)
    angles_deg = torch.as_tensor(degrees_list, dtype=dtype, device=device)

    # 3. Calculate continuous voxel index on GPU
    iso_voxel = (iso_xyz - origin_xyz) / spacing_xyz

    # 4. Handle GPU padding shifts if requested
    if pad_voxels is not None:
        pad_xyz = torch.as_tensor(pad_voxels, dtype=dtype, device=device)
        new_size_xyz = size_xyz + 2 * pad_xyz
        iso_voxel = iso_voxel + pad_xyz

        p_z, p_y, p_x = int(pad_voxels[0]), int(pad_voxels[1]), int(pad_voxels[2])
        tensor_img = F.pad(
            tensor_img, (p_x, p_x, p_y, p_y, p_z, p_z), mode="constant", value=bg_value
        )
        _, _, out_D, out_H, out_W = tensor_img.shape
    else:
        new_size_xyz = size_xyz
        out_D, out_H, out_W = D, H, W

    # 5. Normalize isocentre to PyTorch coordinate space [-1, 1]
    iso_pt = (2.0 * iso_voxel / (new_size_xyz - 1.0)) - 1.0
    x_c, y_c, z_c = iso_pt[0], iso_pt[1], iso_pt[2]

    # 6. Compute trigonometry vectors in parallel for all 180 angles on GPU
    rad = torch.deg2rad(-angles_deg)  # Flip degree to match your operational logic
    cos_a = torch.cos(rad)
    sin_a = torch.sin(rad)

    num_batch = len(degrees_list)
    zeros = torch.zeros(num_batch, device=device)
    ones = torch.ones(num_batch, device=device)

    # 7. Construct the batched 3x4 affine matrices in parallel on GPU
    assert axis == "z"
    row0 = torch.stack([cos_a, -sin_a, zeros, x_c * (1 - cos_a) + y_c * sin_a], dim=1)
    row1 = torch.stack([sin_a, cos_a, zeros, -x_c * sin_a + y_c * (1 - cos_a)], dim=1)
    row2 = torch.stack([zeros, zeros, ones, zeros], dim=1)
    
    # Shape: (180, 3, 4)
    batch_matrices = torch.stack([row0, row1, row2], dim=1).to(dtype)

    # 8. Expand input image and apply the math-shift workaround
    if tensor_img.size(0) != num_batch:
        print(tensor_img.size(), num_batch)
        batched_img = tensor_img.expand(num_batch, -1, -1, -1, -1) - bg_value
    else:
        batched_img = tensor_img - bg_value

    # 9. Execute grid interpolation entirely inside VRAM cores
    grid = F.affine_grid(
        batch_matrices, size=(num_batch, 1, out_D, out_H, out_W), align_corners=True
    )
    rotated_tensor = F.grid_sample(
        batched_img, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )

    # Restore background HU radiation values
    rotated_tensor = rotated_tensor + bg_value

    return rotated_tensor.squeeze(1)


def rotate_pt_z(pt, isocentre, angle):
    """pt, isocentre: LPS from the JSON
    angle: degree to rotate
    returns: the LPS for the rotated pt
    """
    pt = torch.tensor(pt)
    isocentre = torch.tensor(isocentre)
    angle = torch.tensor(angle)

    rad = torch.deg2rad(angle)
    cos_a = torch.cos(rad)
    sin_a = torch.sin(rad)

    x, y, z = pt - isocentre

    R = torch.tensor([[cos_a, -sin_a, 0, 0], [sin_a, cos_a, 0, 0], [0, 0, 1, 0]])

    return torch.matmul(R, torch.tensor([x, y, z, 1])) + isocentre


if __name__ == "__main__":
    # wget https://huggingface.co/datasets/LMUK-RADONC-PHYS-RES/DoseRAD2026/resolve/main/proton/training/1ABB006/dose/Dose_B3_R0_L1.mha
    # wget https://huggingface.co/datasets/LMUK-RADONC-PHYS-RES/DoseRAD2026/resolve/main/proton/training/1ABB006/image/ct.mha
    # wget https://huggingface.co/datasets/LMUK-RADONC-PHYS-RES/DoseRAD2026/resolve/main/proton/training/1ABB006/1ABB006.json

    info = json.load(open("1ABB006.json"))

    # info = {
    #   "iso_center": [-16.1283, -62.0295, 73.9938],
    #   "beams": [
    #     {
    #       "beam_idx": 0,
    #       "gantry_angle": 0.0,
    #       "rays": [
    #         {
    #           "ray_idx": 0,
    #           "ray_source": [-16.13, -1062.03, 33.99],
    #           "ray_target": [-16.13, -62.03, 33.99],
    #           "beamlets": [
    #             {"beamlet_idx": 0, "energy": 45.5978},
    #             {"beamlet_idx": 1, "energy": 126.1016}
    #           ]
    #         }
    #       ]
    #     }
    #   ]
    # }

    isocentre = info["iso_center"]
    degree = -info['beams'][3]['gantry_angle']

    img = sitk.ReadImage(r"ct.mha")
    img_ = sitk.GetImageFromArray(
        rotate_image_batched(
            torch.tensor(sitk.GetArrayFromImage(img)),
            img.GetSpacing(),
            img.GetOrigin(),
            isocentre,
            [degree],
        ).numpy()[0]
    )
    img_.CopyInformation(img)
    sitk.WriteImage(img_, "ct_rotated_30.mha")

    img = sitk.ReadImage(r"Dose_B3_R0_L1.mha")
    img_ = sitk.GetImageFromArray(
        rotate_image_batched(
            torch.tensor(sitk.GetArrayFromImage(img)),
            img.GetSpacing(),
            img.GetOrigin(),
            isocentre,
            [degree],
            bg_value=0
        ).numpy()[0]
    )
    img_.CopyInformation(img)
    sitk.WriteImage(img_, "dose_rotated_30.mha")


    source = info['beams'][3]['rays'][0]['ray_source']
    target = info['beams'][3]['rays'][0]['ray_target']

    src_r = rotate_pt_z(source, isocentre, degree)
    tgt_r = rotate_pt_z(target, isocentre, degree)

    print(f'Source: {source} -> {src_r}')
    print(f'Target: {target} -> {tgt_r}')

    # Source: [483.87, -928.05, 33.99] -> tensor([  -16.1274, -1062.0244,    33.9900])
    # Target: [-14.413875853458421, -61.036056014982364, 33.99] -> tensor([-14.1468, -62.0264,  33.9900])

    # Get the beam path
    import numpy as np
    src_r_ijk = img.TransformPhysicalPointToIndex(src_r.tolist()) # (234, -812, 93)
    tgt_r_ijk = img.TransformPhysicalPointToIndex(tgt_r.tolist()) # (236, 188, 93)

    sx, sy, sz = img.GetSpacing()
    nx, ny, nz = img.GetSize()
    z,y,x = np.mgrid[0:nz, 0:ny, 0:nx]

    centre_x = (src_r_ijk[0] + tgt_r_ijk[0]) / 2
    centre_z = (src_r_ijk[2] + tgt_r_ijk[2]) / 2

    dist = np.sqrt((x-centre_x)**2/sz**2 + (z-centre_z)**2/sx**2)
    img_ = sitk.GetImageFromArray(dist)
    img_.CopyInformation(img)
    sitk.WriteImage(img_, 'dist.nii.gz')
    
    # TODO sigma_s and sigma_e: https://share.google/aimode/nG5QY6Cb1nWiXD3O8
    # sigma_s
    def cal_distance_z(r, sigma_s, alpha):
        sigma_z = np.sqrt(sigma_s**2 + (alpha*z)**2)
        return np.exp(-r**2/(2*sigma_z**2))

    # sigma_e
    from scipy.special import erf
    def cal_fluence_z(z, r0, gamma, sigma_e):
        sigma_r = gamma * sigma_e
        fluence = 0.5 * (1 - erf((z-r0)/np.sqrt(2)*sigma_r))
        return fluence

    def cal_stop_power(z, beta):
        return np.exp(-z*beta)

    # dose deposit (z) = fluence (z) * stop_power (z)