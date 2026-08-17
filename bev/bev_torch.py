import torch
import torch.nn.functional as F
import numpy as np
import cv2
from bev.geometry import get_source_location_mm, get_distance_slice_pt, MLC
import SimpleITK as sitk
import json


def make_grid_5d(B, isocentre_idx, z_scales, ref_img):
    # B = mlc_masks_2d.size(0)                # 180 Control Points
    nz, ny, nx = ref_img.GetSize()
    D, H, W = ny, nx, nz  # Target 3D volume resolution in BEV space

    y_coords = torch.linspace(-1.0, 1.0, H)
    x_coords = torch.linspace(-1.0, 1.0, W)
    Y, X = torch.meshgrid(y_coords, x_coords, indexing='ij')

    isocentre_grid = 2*torch.tensor(isocentre_idx) / torch.tensor([nx,ny,nz]) - 1
    x_c, y_c = isocentre_grid[0], isocentre_grid[2] # Centred at the BEV origin

    X_transformed = (X - x_c) / z_scales + x_c
    Y_transformed = (Y - y_c) / z_scales + y_c
    Z_transformed = torch.zeros_like(X_transformed)  # Constant 0 because input depth is 1

    shared_grid_3d = torch.stack([X_transformed, Y_transformed, Z_transformed], dim=-1) # (90, 128, 128, 3)
    grid_5d = shared_grid_3d.unsqueeze(0).expand(B, -1, -1, -1, -1).float() # [180, 90, 128, 128, 3]

    return grid_5d

def get_bev_torch(mlc_masks_2d, grid_5d):
    input_tensor = mlc_masks_2d.unsqueeze(1).unsqueeze(2)  # Shape: (B, 1, 1, H, W)
    device = grid_5d.device

    beam_masks_3d = F.grid_sample(
        input_tensor.to(device), # [B, 1, 1, nx, nz]
        grid_5d,      # [B, ny, nx, nz, 3]
        mode='nearest',
        padding_mode='zeros',
        align_corners=True
    )

    # Output shape: (B, nx, ny, nz)
    final_bev_volumes = beam_masks_3d.cpu().swapaxes(2,3).squeeze(1).to(torch.uint8) # [B, nx, ny, nz]

    return final_bev_volumes

def cal_scales(ref_img, isocentre, src_mm):
    """Calcalte the ratios relative to distance(iso_slice, source)"""
    isocentre_idx = ref_img.TransformPhysicalPointToIndex(isocentre)

    nz, ny, nx = ref_img.GetSize()
    dist_iso = get_distance_slice_pt(ref_img, isocentre_idx[1], src_mm)
    dist_first = get_distance_slice_pt(ref_img, 0, src_mm)
    dist_last = get_distance_slice_pt(ref_img, ny - 1, src_mm)

    scales = np.linspace(dist_first, dist_last, ny) / dist_iso

    return scales


def mm2idx(ref_img, pts):
    return [ref_img.TransformPhysicalPointToIndex(i) for i in pts]

def draw_iso_mlc(ref_img, mlc):
    x, y, z = ref_img.GetSize()
    arr = np.zeros((z, x), np.uint8)
    shape_idx = [np.array(mm2idx(ref_img, pts)) for pts in mlc]

    cv2_pts = [
        np.array(idx[:, [0, 2]], dtype=np.int32).reshape((-1, 1, 2))
        for idx in shape_idx
    ]

    cv2_bev = cv2.fillPoly(arr, cv2_pts, color=1)
    return cv2_bev

if __name__ == "__main__":
    data_dir = 'data/DoseRAD2026/photon/training/1ABB006/'
    ct = sitk.ReadImage(f'{data_dir}/image/ct.mha')
    beam_info = json.load(open("data/1ABB006.json"))

    # Beam specific
    beam0 = beam_info["beams"][0]
    isocentre = np.array(beam0["iso_center"])
    isocentre_idx = mm2idx(ct, [isocentre])[0]
    sad = beam0["SAD"]
    src_mm = get_source_location_mm(isocentre, 0, sad)
    z_scales = torch.tensor(cal_scales(ct, isocentre, src_mm))[..., None, None]

    # Get the 2d bev at isocentre for all 180 control points
    bev_iso_list = []
    for cp_idx in range(180):
        assert beam0["control_points"][cp_idx]["cp_idx"] == cp_idx
        mlc_left = np.array(beam0["control_points"][cp_idx]["mlc_left_int_mm"])
        mlc_right = np.array(beam0["control_points"][cp_idx]["mlc_right_int_mm"])
        mlc = MLC.get_mlc_segs_mm(mlc_left, mlc_right, isocentre)
        bev_iso = draw_iso_mlc(ct, mlc) # (nx, nz) -> (246, 249)
        bev_iso_list.append(bev_iso)
    bev_iso_list = np.stack(bev_iso_list, axis=0)

    mlc_masks_2d = torch.tensor(bev_iso_list).to(torch.float32)

    # Set the batch number (# of images to process) and get the grid
    # The grid can be reused for each beam
    n_batch = 60
    grid_5d = make_grid_5d(n_batch, isocentre_idx, z_scales).cuda()

    # Get the beam path by batch
    bevs = []
    for i in range(0, 180, n_batch):
        print('Getting BEV:', i, i+n_batch)
        bevs.append(get_bev_torch(mlc_masks_2d[i:(i+n_batch)], grid_5d))
    bevs = torch.cat(bevs, dim=0)


    # Save one as sitk
    cp_idx = 23
    ga = beam0['control_points'][cp_idx]['gantry_angle']
    arr = bevs[cp_idx].numpy()
    img = sitk.GetImageFromArray(arr)
    img.CopyInformation(ct)
    sitk.WriteImage(img, f'bev_{ga}_torch.nii.gz')

    # import matplotlib.pyplot as plt
    # for cp_idx in range(180):
    #     cp_idx = 90
    #     dose = sitk.ReadImage(f'{data_dir}/dose/Dose_B0_CP{cp_idx:03d}.mha')
    #     dose_arr = sitk.GetArrayFromImage(dose)

    #     plt.imshow(dose_arr[:, 139], cmap='gray')
    #     plt.imshow(bevs[cp_idx][:, 139], alpha=0.3)
    #     plt.savefig('test.png')
    #     plt.clf()

    #     break
