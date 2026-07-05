import SimpleITK as sitk
import numpy as np


def make_padded_grid(img, pad_voxels):
    pad_voxels = [50, 50, 50]
    old_size = np.array(img.GetSize())
    old_origin = np.array(img.GetOrigin())
    spacing = np.array(img.GetSpacing())

    new_size = old_size + 2 * np.array(pad_voxels)
    new_size = new_size.tolist()

    new_origin = old_origin - (np.array(pad_voxels) * spacing)
    new_origin = new_origin.tolist()

    return new_size, new_origin


def flat_zeros(img):
    """
    For rotating dose images
    Flat the close to zero values (due to interpolation) to zero
    This will significant reduce the file size
    """
    arr = sitk.GetArrayFromImage(img)
    arr[np.isclose(arr, 0, atol=1e-7)] = 0
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(img)
    return out


def rotate_image(
    infile, isocentre, degree, outfile=None, axis="z", bg_value=-1024, pad_voxels=None
):

    # Based on testing, the degree should be flipped so that
    #   rotate -gantry_angle will return to BEV position
    angle = -degree
    image = sitk.ReadImage(infile)

    if axis == "x":
        radians = np.deg2rad(angle), 0.0, 0.0
    elif axis == "y":
        radians = 0.0, np.deg2rad(angle), 0.0
    elif axis == "z":
        radians = 0.0, 0.0, np.deg2rad(angle)

    transform = sitk.Euler3DTransform()
    transform.SetCenter(isocentre)
    transform.SetRotation(*radians)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(image)
    resampler.SetTransform(transform)
    resampler.SetInterpolator(sitk.sitkBSpline)
    resampler.SetDefaultPixelValue(bg_value)

    if pad_voxels is not None:
        new_size, new_origin = make_padded_grid(image, pad_voxels)
        resampler.SetSize(new_size)
        resampler.SetOutputOrigin(new_origin)
        resampler.SetOutputSpacing(image.GetSpacing())
        resampler.SetOutputDirection(image.GetDirection())

    rotated_image = resampler.Execute(image)

    if outfile is not None:
        sitk.WriteImage(rotated_image, outfile)
    else:
        return rotated_image


if __name__ == "__main__":
    img = rotate_image(
        infile="data/Dose_B0_CP000.mha",
        isocentre=[
            -46.8471844842125,
            27.777663262437926,
            -28.13538836315937,
        ],  # Physical coordinates (x, y, z)
        degree=15,
        bg_value=0,
        # outfile="data/r15p.mha",
        pad_voxels=[50, 50, 50],
    )

    sitk.WriteImage(img, "img1.nii.gz")

    img_flat_zero = flat_zeros(img)
    sitk.WriteImage(img_flat_zero, "img2.nii.gz")
