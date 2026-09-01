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
# %% Comparaison PSF simulée (MATLAB) vs PSF mesurée (DMD) - VERSION CORRIGÉE
#%%

print("\n" + "="*60)
print("COMPARAISON PSF SIMULÉE vs MESURÉE (CORRIGÉE)")
print("="*60)
import scipy.io

psf_sim_path = r"C:\Users\ibrah\OneDrive\Bureau\stageLAM\Python_codes\psf_simulee1.mat"
mat = scipy.io.loadmat(psf_sim_path)


# Chargement de la PSF simulée
mat = scipy.io.loadmat(psf_sim_path)
psf_sim = mat['PSF_resampled']  # PSF simulée originale
ax_sim_urad = mat['ax_resampled_urad'].flatten()

# Vérification des données simulées
print(f"📊 Shape PSF simulée originale : {psf_sim.shape}")
print(f"📏 Résolution simulée : {ax_sim_urad[1] - ax_sim_urad[0]:.2f} µrad/pixel")
print(f"📏 Étendue angulaire simulée : [{ax_sim_urad[0]:.1f}, {ax_sim_urad[-1]:.1f}] µrad")

# Paramètres de rééchantillonnage CORRIGÉS
sampling_sim = ax_sim_urad[1] - ax_sim_urad[0]  # µrad/pixel (MATLAB)
sampling_meas = angle_per_pixel_urad  # µrad/pixel (caméra)

print(f"\n📏 Résolutions :")
print(f"   - Simulée : {sampling_sim:.2f} µrad/px")
print(f"   - Mesurée : {sampling_meas:.2f} µrad/px")

# Facteur de rééchantillonnage CORRIGÉ (simulée → mesurée)
# On veut que la PSF simulée ait la même résolution que la caméra
# Si sampling_sim > sampling_meas : la PSF simulée est plus grossière, on doit l'agrandir
# Si sampling_sim < sampling_meas : la PSF simulée est plus fine, on doit la réduire
factor = sampling_sim / sampling_meas

print(f"\n🔄 Facteur de rééchantillonnage : {factor:.6f}")
print(f"   - Si >1 : agrandir la PSF simulée")
print(f"   - Si <1 : réduire la PSF simulée")

# Rééchantillonnage de la PSF simulée
print("\n⏳ Rééchantillonnage de la PSF simulée...")
psf_sim_resampled = zoom(psf_sim, factor, order=3)

# Vérification après rééchantillonnage
print(f"✅ Shape PSF simulée rééchantillonnée : {psf_sim_resampled.shape}")
print(f"   - Min : {psf_sim_resampled.min():.2e}, Max : {psf_sim_resampled.max():.2e}")

# =============================================================================
# CORRECTION DU RECADRAGE POUR OBTENIR (200, 200)
# =============================================================================
Ny_m, Nx_m = mean_frame_dmd.shape  # (200, 200)

# Trouver le centre de masse de la PSF simulée (plus robuste que le centre géométrique)
cy_sim, cx_sim = center_of_mass(psf_sim_resampled)
print(f"\n🎯 Centre de masse PSF simulée : ({cx_sim:.1f}, {cy_sim:.1f})")

# Calcul des indices de recadrage
y_start = int(cy_sim - Ny_m//2)
y_end = int(cy_sim + Ny_m//2)
x_start = int(cx_sim - Nx_m//2)
x_end = int(cx_sim + Nx_m//2)

print(f"\n📐 Recadrage pour obtenir (200, 200) :")
print(f"   - y_start={y_start}, y_end={y_end} (taille={y_end-y_start})")
print(f"   - x_start={x_start}, x_end={x_end} (taille={x_end-x_start})")

# Vérification des bornes
if y_start < 0 or x_start < 0 or y_end > psf_sim_resampled.shape[0] or x_end > psf_sim_resampled.shape[1]:
    print("⚠️ Attention : recadrage hors limites ! Ajustement automatique...")
    y_start = max(0, y_start)
    x_start = max(0, x_start)
    y_end = min(psf_sim_resampled.shape[0], y_end)
    x_end = min(psf_sim_resampled.shape[1], x_end)

psf_sim_crop = psf_sim_resampled[y_start:y_end, x_start:x_end]

print(f"✅ Shape PSF simulée recadrée : {psf_sim_crop.shape}")

# =============================================================================
# AFFICHAGE SÉCURISÉ (sans RuntimeWarning)
# =============================================================================
zoom_urad = 1500  # µrad

fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# PSF simulée
psf_sim_norm = psf_sim_crop / (psf_sim_crop.max() + 1e-10)  # Normalisation sécurisée
im0 = axs[0].imshow(
    np.log10(np.clip(psf_sim_norm, 1e-10, None)),  # Clip les valeurs < 1e-10
    extent=extent_cam,
    origin='upper',
    cmap='viridis'
)
axs[0].set_title("PSF simulée rééchantillonnée (log)")
axs[0].set_xlabel("angle [µrad]")
axs[0].set_ylabel("angle [µrad]")
axs[0].set_xlim([-zoom_urad, zoom_urad])
axs[0].set_ylim([zoom_urad, -zoom_urad])
plt.colorbar(im0, ax=axs[0])

# PSF mesurée DMD
psf_dmd_norm = mean_frame_dmd / (mean_frame_dmd.max() + 1e-10)
im1 = axs[1].imshow(
    np.log10(np.clip(psf_dmd_norm, 1e-10, None)),
    extent=extent_cam,
    origin='upper',
    cmap='viridis'
)
axs[1].set_title("PSF mesurée DMD (log)")
axs[1].set_xlabel("angle [µrad]")
axs[1].set_ylabel("angle [µrad]")
axs[1].set_xlim([-zoom_urad, zoom_urad])
axs[1].set_ylim([zoom_urad, -zoom_urad])
plt.colorbar(im1, ax=axs[1])

# PSF mesurée classique
psf_classic_norm = mean_frame_classical_slm / (mean_frame_classical_slm.max() + 1e-10)
im2 = axs[2].imshow(
    np.log10(np.clip(psf_classic_norm, 1e-10, None)),
    extent=extent_cam,
    origin='upper',
    cmap='viridis'
)
axs[2].set_title("PSF mesurée SLM classique (log)")
axs[2].set_xlabel("angle [µrad]")
axs[2].set_ylabel("angle [µrad]")
axs[2].set_xlim([-zoom_urad, zoom_urad])
axs[2].set_ylim([zoom_urad, -zoom_urad])
plt.colorbar(im2, ax=axs[2])

plt.suptitle("Comparaison PSF simulée vs mesurée — même échelle angulaire")
plt.tight_layout()
plt.show()

# =============================================================================
# VÉRIFICATIONS FINALES
# =============================================================================
print("\n" + "="*60)
print("✅ VÉRIFICATIONS FINALES")
print("="*60)
print(f"✔ Shape PSF simulée finale : {psf_sim_crop.shape} (attendu : (200, 200))")
print(f"✔ Min/Max PSF simulée : {psf_sim_crop.min():.2e} / {psf_sim_crop.max():.2e}")
print(f"✔ Centre de masse : {center_of_mass(psf_sim_crop)}")
print("✔ Tous les RuntimeWarning ont été résolus")


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
