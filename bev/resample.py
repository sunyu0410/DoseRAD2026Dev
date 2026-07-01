import SimpleITK as sitk


def resample(image, new_spacing=(1, 1, 1), bg_value=-1024, is_label=False):

    # 1. Get current image attributes
    original_size = image.GetSize()
    original_spacing = image.GetSpacing()  # Assumed to be (2.0, 2.0, 2.0)

    # 2. Calculate the new grid size
    # For (2,2,2) to (1,1,1), this will multiply each dimension by 2
    new_size = [
        int(round(original_size[i] * original_spacing[i] / new_spacing[i]))
        for i in range(image.GetDimension())
    ]

    # 3. Select the correct interpolator based on image type
    # - sitkLinear for intensity images (CT, MRI)
    # - sitkNearestNeighbor for binary/segmentation masks to avoid blurring borders
    interpolator = sitk.sitkNearestNeighbor if is_label else sitk.sitkBSpline

    # 4. Run the resample function using an identity transform
    resampled_image = sitk.Resample(
        image,
        new_size,
        sitk.Transform(),  # Identity transform keeps physical location locked
        interpolator,
        image.GetOrigin(),
        new_spacing,
        image.GetDirection(),
        bg_value,  # Default pixel value for padding if needed
        image.GetPixelID(),
    )
    return resampled_image


if __name__ == "__main__":
    img_2mm = sitk.ReadImage("data/ct.mha")
    img_1mm = resample(img_2mm, new_spacing=[1.0, 1.0, 1.0], is_label=False)
    print(img_1mm.GetSpacing(), img_1mm.GetSize(), img_1mm.GetOrigin())
    sitk.WriteImage(img_1mm, "data/ct_1mm.mha")
