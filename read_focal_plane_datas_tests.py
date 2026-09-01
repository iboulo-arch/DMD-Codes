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
colorbar = plt.colorbar(axs[0].imshow(mean_reference_psf), ax=axs[0])
axs[1].imshow(mean_frame_classical_slm)
axs[1].set_title("PSF (mode classique SLM)")
colorbar = plt.colorbar(axs[1].imshow(mean_frame_classical_slm), ax=axs[1])
axs[2].imshow(mean_frame_dmd)
axs[2].set_title("PSF (mode DMD)")
colorbar = plt.colorbar(axs[2].imshow(mean_frame_dmd), ax=axs[2])

# -------------------------my added code-------------------------
# %% Comparaison des PSF : diff, ratio, 

diff_norm = np.abs(mean_frame_classical_slm / mean_frame_classical_slm.max()
                    - mean_frame_dmd / mean_frame_dmd.max())
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
im = ax.imshow(diff_norm)
ax.set_title("Différence normalisée ('slm_classique' - 'slm_DMD')")
plt.colorbar(im, ax=ax)

plt.show()


# %% affichage échelle log
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
for ax, img, title in zip(axs, [mean_reference_psf, mean_frame_classical_slm, mean_frame_dmd],
                           ["référence", "classique", "DMD"]):
    im = ax.imshow(np.log10(np.clip(img, 1, None)))
    ax.set_title(f"PSF {title} (log)")
    plt.colorbar(im, ax=ax)
plt.show()


# %%comaparaison des énergies
print("Pic classique :", mean_frame_classical_slm.max())
print("Pic DMD :", mean_frame_dmd.max())
print("Ratio des pics (DMD/classique) :", mean_frame_dmd.max() / mean_frame_classical_slm.max())

print("Energie totale référence :", mean_reference_psf.sum())
print("Énergie totale classique :", mean_frame_classical_slm.sum())
print("Énergie totale DMD :", mean_frame_dmd.sum())
print("Ratio dmd/classique:", mean_frame_dmd.sum() / mean_frame_classical_slm.sum())


# %% Corrélation croisée entre les PSF
from scipy.signal import correlate2d

corr = correlate2d(mean_frame_classical_slm, mean_frame_dmd, mode='same')
print("Corrélation max normalisée :", corr.max() / np.sqrt((mean_frame_classical_slm**2).sum() * (mean_frame_dmd**2).sum()))
# %% Profils radiaux des PSF
from scipy.ndimage import center_of_mass

def radial_profile(img):
    cy, cx = center_of_mass(img)
    y, x = np.indices(img.shape)
    r = np.sqrt((x - cx)**2 + (y - cy)**2).astype(int)
    profile = np.bincount(r.ravel(), img.ravel()) / np.bincount(r.ravel())
    return profile

plt.figure()
plt.plot(radial_profile(mean_reference_psf), label="référence")
plt.plot(radial_profile(mean_frame_classical_slm), label="classique")
plt.plot(radial_profile(mean_frame_dmd), label="DMD")
plt.yscale("log")
plt.xlabel("rayon (px)")
plt.ylabel("intensité moyenne")
plt.legend()
plt.title("Profils radiaux des PSF")

# %%
