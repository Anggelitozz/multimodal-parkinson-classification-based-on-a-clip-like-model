import nibabel as nib
from pathlib import Path

images_dir = Path("/home/franklin_pupils/franklin_pupils/angel/SAM-Med3D-main/data/train_test/harvard_oxford_atlas/SPECT/labelsTr")

for nii_file in images_dir.glob("*.nii"):
    img = nib.load(str(nii_file))
    out_file = nii_file.with_suffix(".nii.gz")
    nib.save(img, str(out_file))
    nii_file.unlink()  # elimina el .nii original
    print(f"Convertido: {nii_file} -> {out_file}")
