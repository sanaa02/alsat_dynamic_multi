#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
from pyhdf.SD import SD

data_dir = "data/modis_patches"

files = sorted(
    [f for f in os.listdir(data_dir)
     if f.endswith(".hdf")]
)

if len(files) == 0:
    raise ValueError("No HDF files found")

print(f"Found {len(files)} files")

for idx, file in enumerate(files):

    path = os.path.join(data_dir, file)

    try:
        print(f"\n[{idx+1}/{len(files)}] Loading {file}")

        sd = SD(path)

        # RGB bands
        b1 = sd.select("sur_refl_b01_1")[:].astype(np.float32)
        b4 = sd.select("sur_refl_b04_1")[:].astype(np.float32)
        b3 = sd.select("sur_refl_b03_1")[:].astype(np.float32)

        for b in [b1, b4, b3]:
            b[b == -28672] = 0
            b *= 0.0001

        # brighten image
        rgb = np.stack([
            np.clip(b1 * 3, 0, 1),
            np.clip(b4 * 3, 0, 1),
            np.clip(b3 * 3, 0, 1)
        ], axis=-1)

        qa = sd.select("state_1km_1")[:]
        cloud = qa & 3

        sd.end()

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(18,8)
        )

        axes[0].imshow(rgb)
        axes[0].set_title("RGB")
        axes[0].axis("off")

        axes[1].imshow(cloud)
        axes[1].set_title("Cloud Mask")
        axes[1].axis("off")

        fig.suptitle(
            f"{file}\nClose window for next image",
            fontsize=14
        )

        plt.show()

    except Exception as e:
        print(f"Failed: {file}")
        print(e)

print("\nFinished displaying all files.")