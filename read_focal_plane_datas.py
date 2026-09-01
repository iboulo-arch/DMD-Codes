# %%

import h5py
import matplotlib.pyplot as plt
import numpy as np

focal_plane_file_path = r"C:\Users\ibrah\OneDrive\Bureau\stageLAM\Python_codes\utc_2026-06-17_09-49-51_test_focal_plane_camera.h5"

with h5py.File(
    focal_plane_file_path,
    mode="r",
) as f:
    exposure_time = f.attrs["exposure_time"]
    n_frames_avg = f.attrs["n_frames_avg"]

    command_slm_classic = f["command_slm_classic"][...]
    command_slm_dmd = f["command_slm_dmd"][...]
    dark = f["dark"][...]
    mean_reference_psf = f["mean_reference_psf"][...]
    mean_frame_classical_slm = f["mean_frame_classical_slm"][...]
    mean_frame_dmd = f["mean_frame_dmd"][...]

print(f"exposure_time = {exposure_time} s, n_frames_avg = {n_frames_avg}")

# %% Affichage des commandes envoyées au SLM

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(command_slm_classic)
axs[0].set_title("Commande SLM classique")
axs[1].imshow(command_slm_dmd)
axs[1].set_title("Commande SLM (DMD)")

# %% Affichage du dark et de la PSF de référence

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(dark)
axs[0].set_title("Dark")
axs[1].imshow(mean_reference_psf)
axs[1].set_title("PSF de référence")

# %% Comparaison des PSF : classique vs DMD

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(mean_reference_psf)
axs[0].set_title("PSF référence")
axs[1].imshow(mean_frame_classical_slm)
axs[1].set_title("PSF (mode classique)")
axs[2].imshow(mean_frame_dmd)
axs[2].set_title("PSF (mode DMD)")

# %% Comparaison des PSF : diff, ratio, 

ratio_psf =  mean_frame_dmd/mean_reference_psf
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
im = ax.imshow(ratio_psf, vmin=0, vmax=1)
ax.set_title("Ratio PSF (DMD / ref)")
plt.colorbar(im, ax=ax)

plt.show()

# %%