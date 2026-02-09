# revisar_metadata_nifti.py

import nibabel as nib
import sys
from pathlib import Path

def revisar_metadata(nii_path):
    nii = nib.load(nii_path)
    header = nii.header

    print(f"\n Archivo: {nii_path}")
    print("-" * 60)
    print(f"Dimensiones (shape): {nii.shape}")
    print(f"Número de dimensiones: {len(nii.shape)}")
    print(f"Tamaño de voxel (pixdim): {header.get_zooms()}")
    print(f"Tipo de dato: {header.get_data_dtype()}")
    print(f"Affine matrix:\n{nii.affine}")
    print(f"Descripciones extras en el header:\n{header}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python revisar_metadata_nifti.py archivo1.nii [archivo2.nii.gz ...]")
        sys.exit(1)

    for f in sys.argv[1:]:
        path = Path(f)
        if not path.exists():
            print(f"no se encontró el archivo: {f}")
        else:
            revisar_metadata(f)
