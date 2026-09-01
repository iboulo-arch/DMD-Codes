# %% ============================================================
# Monitoring live du rapport de Strehl - ThorCam CS165MU1(/M)
# ==================================================================
#
# Principe :
#   Strehl_ratio ~= (I_pic_mesure / Flux_total_mesure) / (I_pic_ref / Flux_total_ref)
#
# Deux modes de reference possibles (choisir REF_MODE plus bas) :
#   - "capture"    : on capture une PSF de reference au debut du script
#                    (ex: DMD a plat / meilleure correction connue),
#                    et on compare tout le reste a cette capture.
#                    -> Strehl RELATIF, tres robuste, recommande pour
#                       comparer rapidement des formes de wavefront.
#   - "theorique"  : on simule la PSF ideale (limitee par diffraction)
#                    a partir de la pupille elliptique via FFT, et on
#                    compare la mesure a ce cas ideal.
#                    -> Strehl plus proche de l'absolu, mais suppose
#                       une bonne calibration photometrique
#                       (flat field, pas de saturation, meme expo).
#
# Pre-requis :
#   pip install thorlabs_tsi_sdk numpy matplotlib
#   + Thorlabs Camera SDK installe (DLLs), cf. THORLABS_DLL_PATH plus bas.
#
# Notes materiel CS165MU1 :
#   - Capteur Sony IMX250, mono, 12 bits -> valeurs ADU dans [0, 4095]
#   - Pixel size ~3.45 um
#   - Attention a la saturation : verifier frame.max() < 4095 en permanence,
#     sinon le rapport de Strehl mesure est totalement faux.
# ==================================================================

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

# %% ============================================================
# Configuration
# ==================================================================

# Chemin vers les DLLs du SDK Thorlabs (a adapter a ton installation,
# typiquement dans "Scientific Camera Interfaces/SDK/Python Toolkit/dlls/64_lib")
THORLABS_DLL_PATH = r"C:\Program Files\Thorlabs\ThorImageCAM\Bin"

EXPOSURE_TIME_US = 11000      # a ajuster : pas de saturation, bon SNR
ROI_HALF_SIZE = 60           # demi-taille (en pixels) de la zone autour du pic PSF
BACKGROUND_LEVEL = 0         # offset moyen du capteur si mesure au prealable (obturateur ferme)
SATURATION_ADU = 4000        # marge de securite avant 4095 (12 bits)

REF_MODE = "theorique"         # "capture" ou "theorique"

# Parametres pour le mode "theorique" (pupille elliptique, cf. maoppy)
WAVELENGTH_NM = 635.0
PUPIL_A_PIX = 400            # demi-grand axe pupille (meme convention que make_elliptical_mask)
PUPIL_B_PIX = 320            # demi-petit axe pupille
PUPIL_THETA_DEG = -135.0     # orientation pupille

HISTORY_LEN = 300            # nombre de points affiches dans la courbe temporelle


# %% ============================================================
# Connexion camera
# ==================================================================

def configure_windows_dll_path(dll_path):
    """A appeler avant l'import de thorlabs_tsi_sdk sur Windows."""
    if os.path.isdir(dll_path):
        os.add_dll_directory(dll_path)
    else:
        print(f"[ATTENTION] Chemin DLL introuvable : {dll_path}")


configure_windows_dll_path(THORLABS_DLL_PATH)

from thorlabs_tsi_sdk.tl_camera import TLCameraSDK  # noqa: E402


def open_camera(sdk):
    cameras = sdk.discover_available_cameras()
    if not cameras:
        raise RuntimeError("Aucune camera Thorlabs detectee.")
    camera = sdk.open_camera(cameras[0])
    camera.exposure_time_us = EXPOSURE_TIME_US
    camera.frames_per_trigger_zero_for_unlimited = 0  # mode continu
    camera.image_poll_timeout_ms = 1000
    camera.arm(2)
    camera.issue_software_trigger()
    return camera


def grab_frame(camera):
    """Recupere la derniere frame disponible, en numpy array 2D (uint16)."""
    frame = camera.get_pending_frame_or_null()
    if frame is None:
        return None
    image = np.copy(frame.image_buffer).astype(np.float64)
    return image


# %% ============================================================
# Calcul du rapport pic / flux (mesure)
# ==================================================================

def crop_around_peak(image, half_size=ROI_HALF_SIZE):
    """Recadre une petite fenetre centree sur le pixel le plus intense."""
    iy, ix = np.unravel_index(np.argmax(image), image.shape)
    y0, y1 = max(0, iy - half_size), min(image.shape[0], iy + half_size)
    x0, x1 = max(0, ix - half_size), min(image.shape[1], ix + half_size)
    return image[y0:y1, x0:x1]


def peak_to_flux_ratio(image, background=BACKGROUND_LEVEL):
    """
    Calcule I_pic / Flux_total sur une ROI centree sur le pic.
    C'est cette quantite qui, normalisee par une reference, donne le Strehl.
    """
    img = image - background
    img[img < 0] = 0

    roi = crop_around_peak(img)

    peak = roi.max()
    flux = roi.sum()

    if peak >= SATURATION_ADU:
        print("[ATTENTION] Pixel proche saturation ! Baisser EXPOSURE_TIME_US.")

    if flux <= 0:
        return 0.0
    return peak / flux


# %% ============================================================
# PSF theorique (mode "theorique") - pupille elliptique via FFT
# ==================================================================

def make_elliptical_mask(shape, a, b, theta_deg, center=None):
    ny, nx = shape
    if center is None:
        center = (ny // 2, nx // 2)
    yy, xx = np.mgrid[0:ny, 0:nx]
    y = yy - center[0]
    x = xx - center[1]
    theta = np.deg2rad(theta_deg)
    xr = x * np.cos(theta) + y * np.sin(theta)
    yr = -x * np.sin(theta) + y * np.cos(theta)
    return ((xr / a) ** 2 + (yr / b) ** 2) <= 1.0


def theoretical_peak_to_flux_ratio(shape=(1024, 1024)):
    """
    Simule la PSF limitee par diffraction d'une pupille elliptique parfaite
    (phase nulle), et renvoie son ratio pic/flux comme reference "Strehl = 1".
    """
    pupil = make_elliptical_mask(shape, PUPIL_A_PIX, PUPIL_B_PIX, PUPIL_THETA_DEG)
    field = pupil.astype(np.complex128)
    psf = np.abs(np.fft.fftshift(np.fft.fft2(field))) ** 2
    peak = psf.max()
    flux = psf.sum()
    return peak / flux


# %% ============================================================
# Boucle live
# ==================================================================

def main():
    with TLCameraSDK() as sdk:
        camera = open_camera(sdk)
        print("Camera connectee. Acquisition en cours...")

        # --- Determination de la reference ---
        if REF_MODE == "capture":
            input("Mets le DMD dans l'etat de reference (ex: a plat), puis Entree...")
            ref_frame = grab_frame(camera)
            while ref_frame is None:
                ref_frame = grab_frame(camera)
            ref_ratio = peak_to_flux_ratio(ref_frame)
            print(f"Reference capturee. ratio_ref = {ref_ratio:.5f}")
        elif REF_MODE == "theorique":
            ref_ratio = theoretical_peak_to_flux_ratio()
            print(f"Reference theorique calculee. ratio_ref = {ref_ratio:.5f}")
        else:
            raise ValueError("REF_MODE doit etre 'capture' ou 'theorique'")

        # --- Affichage live ---
        plt.ion()
        fig, (ax_img, ax_curve) = plt.subplots(1, 2, figsize=(10, 4.5))

        first_frame = None
        while first_frame is None:
            first_frame = grab_frame(camera)
        roi0 = crop_around_peak(first_frame)
        im_disp = ax_img.imshow(roi0, cmap="inferno")
        ax_img.set_title("PSF (ROI)")
        plt.colorbar(im_disp, ax=ax_img, fraction=0.046)

        strehl_history = []
        (line,) = ax_curve.plot([], [], "-o", ms=3)
        ax_curve.set_xlabel("frame")
        ax_curve.set_ylabel("Strehl estime")
        ax_curve.set_ylim(0, 1.1)
        ax_curve.grid(alpha=0.3)

        try:
            while True:
                frame = grab_frame(camera)
                if frame is None:
                    time.sleep(0.01)
                    continue

                ratio = peak_to_flux_ratio(frame)
                strehl = ratio / ref_ratio

                strehl_history.append(strehl)
                if len(strehl_history) > HISTORY_LEN:
                    strehl_history.pop(0)

                im_disp.set_data(crop_around_peak(frame))
                im_disp.set_clim(0, max(1, crop_around_peak(frame).max()))

                line.set_data(range(len(strehl_history)), strehl_history)
                ax_curve.set_xlim(0, max(HISTORY_LEN, len(strehl_history)))
                ax_curve.set_title(f"Strehl = {strehl:.3f}")

                fig.canvas.draw_idle()
                plt.pause(0.03)

        except KeyboardInterrupt:
            print("Arret demande par l'utilisateur.")
        finally:
            camera.disarm()
            camera.dispose()


if __name__ == "__main__":
    main()