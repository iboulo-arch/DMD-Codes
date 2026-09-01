# %%

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import correlate2d


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

#------------section pour comparer psf simul et psf mesurée en passant aux coordonnées angulaires (µrad)----------------
# Conversion pixels caméra → µrad
pitch_camera = 3.45e-6   # m
f_L4bis      = 200e-3    # m

angle_per_pixel_camera = pitch_camera / f_L4bis  # rad/pixel
angle_per_pixel_camera_urad = angle_per_pixel_camera * 1e6  # µrad/pixel

Ny, Nx = mean_reference_psf.shape
extent = [
    -(Nx//2) * angle_per_pixel_camera_urad,
     (Nx//2) * angle_per_pixel_camera_urad,
     (Ny//2) * angle_per_pixel_camera_urad,
    -(Ny//2) * angle_per_pixel_camera_urad,
]

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
for ax, img, title in zip(axs,
    [mean_reference_psf, mean_frame_classical_slm, mean_frame_dmd],
    ["PSF référence", "PSF classique SLM", "PSF DMD"]):
    im = ax.imshow(img, extent=extent)
    ax.set_title(title)
    ax.set_xlabel("angle [µrad]")
    ax.set_ylabel("angle [µrad]")
    plt.colorbar(im, ax=ax)
# -------------------------my added code-------------------------

#%%Verification du PSF plat en faisant la |FFT(image)|^2

psf_verif = np.abs(np.fft.fftshift(np.fft.fft2(mean_reference_psf)))**2

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
im = ax.imshow(np.log10(psf_verif / psf_verif.max()+ 1e-6))
ax.set_title("PSF plat (log)")
plt.colorbar(im, ax=ax)


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



# %% Comparaison PSF simulée (MATLAB) vs PSF mesurée (DMD)
import scipy.io

psf_sim_path = r"C:\Users\ibrah\OneDrive\Bureau\stageLAM\Python_codes\psf_simulee1.mat"
mat = scipy.io.loadmat(psf_sim_path)
psf_sim     = mat['PSF_resampled']
ax_sim_urad = mat['ax_resampled_urad'].flatten()

extent_sim = [ax_sim_urad[0], ax_sim_urad[-1],
              ax_sim_urad[-1], ax_sim_urad[0]]

# Axes caméra (déjà calculés plus haut)
pitch_camera = 3.45e-6
f_L4bis      = 200e-3
angle_per_pixel_urad = pitch_camera / f_L4bis * 1e6
Ny, Nx = mean_frame_dmd.shape
extent_cam = [
    -(Nx//2) * angle_per_pixel_urad,
     (Nx//2) * angle_per_pixel_urad,
     (Ny//2) * angle_per_pixel_urad,
    -(Ny//2) * angle_per_pixel_urad,
]

extent_sim = [ax_sim_urad[0], ax_sim_urad[-1],
              ax_sim_urad[-1], ax_sim_urad[0]]

from scipy.ndimage import zoom

# Facteur de rééchantillonnage
sampling_sim  = 86.0    # µrad/pixel  (MATLAB)
sampling_meas = 17.25   # µrad/pixel  (caméra)
factor = sampling_sim / sampling_meas  # ≈ 4.986
#factor = 6

# Zoom de la PSF simulée → même résolution que la caméra
psf_sim_resampled = zoom(psf_sim, factor, order=1)
#psf_sim_resampled = psf_sim

# Recadrage centré pour obtenir (200, 200)
Ny_m, Nx_m = mean_frame_dmd.shape  # (200, 200)
cy, cx = np.array(psf_sim_resampled.shape) // 2  # centre géométrique (256, 256)

psf_sim_crop = psf_sim_resampled[
    cy - Ny_m//2 : cy + Ny_m//2,   # 256-100 : 256+100 → (200, 200) ✅
    cx - Nx_m//2 : cx + Nx_m//2
]

print("Shape crop :", psf_sim_crop.shape)  # → (200, 200) ✅


# Vérification
print("Shape simulée rééch. :", psf_sim_resampled.shape)
print("Shape crop            :", psf_sim_crop.shape)
print("Shape mesurée         :", mean_frame_dmd.shape)

# Affichage comparatif sur la même échelle angulaire
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

axs[0].imshow(np.log10(psf_sim_crop / psf_sim_crop.max() + 1e-6),
              extent=extent_cam, origin='upper')
axs[0].set_title("PSF simulée rééchantillonnée (log)")
axs[0].set_xlabel("angle [µrad]"); axs[0].set_ylabel("angle [µrad]")
# Définir la valeur de zoom_urad utilisée pour les limites d'axe
zoom_urad = 1500  # µrad
axs[0].set_xlim([-zoom_urad, zoom_urad]); axs[0].set_ylim([zoom_urad, -zoom_urad])

axs[1].imshow(np.log10(mean_frame_dmd / mean_frame_dmd.max() + 1e-6),
              extent=extent_cam, origin='upper')
axs[1].set_title("PSF mesurée DMD (log)")
axs[1].set_xlabel("angle [µrad]"); axs[1].set_ylabel("angle [µrad]")
axs[1].set_xlim([-zoom_urad, zoom_urad]); axs[1].set_ylim([zoom_urad, -zoom_urad])

# ajouter le slm clissique pour comparaison
axs[2].imshow(np.log10(mean_frame_classical_slm / mean_frame_classical_slm.max() + 1e-6),
              extent=extent_cam, origin='upper')
axs[2].set_title("PSF mesurée SLM classique (log)")
axs[2].set_xlabel("angle [µrad]"); axs[2].set_ylabel("angle [µrad]")
axs[2].set_xlim([-zoom_urad, zoom_urad]); axs[2].set_ylim([zoom_urad, -zoom_urad])

plt.suptitle("Comparaison PSF simulée vs mesurée — même échelle angulaire")
plt.tight_layout(); plt.show()

# Corrélation croisée sur grille commune
corr = correlate2d(psf_sim_crop / psf_sim_crop.max(),
                   mean_frame_dmd / mean_frame_dmd.max(), mode='same')
print("Corrélation max normalisée :", corr.max() /
      np.sqrt((psf_sim_crop**2).sum() * (mean_frame_dmd**2).sum()))



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
plt.plot(radial_profile(psf_sim_crop), label="simulée")
plt.yscale("log")
plt.xlabel("rayon (px)")
plt.ylabel("intensité moyenne")
plt.legend()
plt.title("Profils radiaux des PSF")


# %%
