import SimpleITK as sitk
import os
from tqdm import tqdm

# On the photon data
walk = os.walk('data/DoseRAD2026/photon')
info = []

for (_dir, _folders, _files) in walk:
    # Sample every 10 files
    for f in tqdm(_files[::10]):
        if f.endswith('.mha'):
            img = sitk.ReadImage(os.path.join(_dir, f))
            info.append((os.path.join(_dir, f), sitk.GetArrayFromImage(img).shape))

    print(len(info))