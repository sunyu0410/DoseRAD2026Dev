import SimpleITK as sitk
import numpy as np


def rotate_image(infile, outfile, isocentre, degree, axis="z", bg_value=-1024):
    image = sitk.ReadImage(infile)

    if axis == "x":
        radians = np.deg2rad(degree), 0.0, 0.0
    elif axis == "y":
        radians = 0.0, np.deg2rad(degree), 0.0
    elif axis == "z":
        radians = 0.0, 0.0, np.deg2rad(degree)

    transform = sitk.Euler3DTransform()
    transform.SetCenter(isocentre)
    transform.SetRotation(*radians)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(image)
    resampler.SetTransform(transform)
    resampler.SetInterpolator(sitk.sitkBSpline)
    resampler.SetDefaultPixelValue(bg_value)
    rotated_image = resampler.Execute(image)

    sitk.WriteImage(rotated_image, outfile)


if __name__ == "__main__":
    rotate_image(
        infile="data/r15.mha",
        outfile="data/r0.mha",
        isocentre=[
            46.8471844842125,
            -27.777663262437926,
            -28.13538836315937,
        ],  # Physical coordinates (x, y, z)
        degree=-15,
    )
