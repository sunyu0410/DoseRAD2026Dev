import torch
import torch.nn.functional as F
import SimpleITK as sitk
from time import perf_counter


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
        tensor_img = tensor_img.unsqueeze(0)

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
    if axis == "z":
        row0 = torch.stack([cos_a, -sin_a, zeros, x_c * (1 - cos_a) + y_c * sin_a], dim=1)
        row1 = torch.stack([sin_a, cos_a, zeros, -x_c * sin_a + y_c * (1 - cos_a)], dim=1)
        row2 = torch.stack([zeros, zeros, ones, zeros], dim=1)
    elif axis == "y":
        row0 = torch.stack([cos_a, zeros, sin_a, x_c * (1 - cos_a) - z_c * sin_a], dim=1)
        row1 = torch.stack([zeros, ones, zeros, zeros], dim=1)
        row2 = torch.stack([-sin_a, zeros, cos_a, x_c * sin_a + z_c * (1 - cos_a)], dim=1)
    elif axis == "x":
        row0 = torch.stack([ones, zeros, zeros, zeros], dim=1)
        row1 = torch.stack([zeros, cos_a, -sin_a, y_c * (1 - cos_a) + z_c * sin_a], dim=1)
        row2 = torch.stack([zeros, sin_a, cos_a, -y_c * sin_a + z_c * (1 - cos_a)], dim=1)

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



def rotate_image_split(
    tensor_img,  # Shape: (D, H, W) or (1, 1, D, H, W) on GPU
    spacing,  # Can now be a torch.Tensor or list/tuple
    origin,  # Can now be a torch.Tensor or list/tuple
    isocentre,  # Can now be a torch.Tensor or list/tuple
    degrees_list,  # Can now be a torch.Tensor of angles on GPU
    axis="z",
    bg_value=-1024,
    pad_voxels=None,
    split_size = 6,
):
    
    """
    Perform the same operation as rotate_image_batched but split the input into smaller trunks
    """
    result = []
    
    for degrees in torch.split(degrees_list, split_size_or_sections=split_size):
        img_rot = rotate_image_batched(
            tensor_img,
            spacing,
            origin,
            isocentre,
            degrees,
            axis,
            bg_value,
            pad_voxels,
        )
        result.append(img_rot)

    del img_rot
    torch.cuda.empty_cache()
    return torch.cat(result, dim=0)
    


if __name__ == "__main__":
    img = sitk.ReadImage("data/DoseRAD2026/photon/training/1ABB006/image/ct.mha")
    img_arr = sitk.GetArrayFromImage(img)
    img_tensor = torch.tensor(img_arr).to(torch.float16).cuda()
    isocentre = [-46.8471844842125, 27.777663262437926, -8.135388363159374]

    torch.cuda.empty_cache()
    start = perf_counter()
    degrees = range(20)
    result = rotate_image_split(img_tensor, img.GetSpacing(), img.GetOrigin(), isocentre, degrees)
    end = perf_counter()

    print(end - start)
