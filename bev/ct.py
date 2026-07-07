import SimpleITK as sitk


def get_body_mask(img_sitk, thres=-1024):
    return img_sitk > thres


if __name__ == "__main__":
    img = sitk.ReadImage("data/ct.mha")
    mask = get_body_mask(img)
    sitk.WriteImage(mask, "data/body_mask.nii.gz")
