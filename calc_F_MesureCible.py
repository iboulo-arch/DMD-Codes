# -*- coding: utf-8 -*-
"""
calc_fidelity.py

Reprend calc_plots.py et ajoute le calcul de la fidelite F entre :
  - la phase mesuree par le HASO (mode_inj - flat = "Reel mode_inj")
  - le mode de Zernike theorique correspondant, genere avec la
    VRAIE librairie maoppy (maoppy.zernike : zernike, ansi2nm,
    ansi_name), exactement comme dans DMD_reconstruction_V2.py.

    /!\ maoppy.ansi2nm() NE SUIT PAS l'indexation Noll standard
    au-dela des premiers modes (ex: j=40 -> maoppy donne m=0,
    alors que Noll standard donnerait m=4). Il faut donc utiliser
    maoppy directement plutot qu'une reimplementation, pour rester
    coherent avec les modes reellement envoyes au DM/SLM.

Definition de fidelite utilisee (recouvrement de champ complexe,
identique a celle du script super_pixel_DMD.py) :

    Et = exp(i*Phi_cible)      (mode de Zernike theorique, normalise a
                                 la meme RMS que la mesure pour une
                                 comparaison honnete)
    Er = exp(i*Phi_mesuree)    (mode reellement injecte, mesure au HASO)

    F = | <Et|Er> |^2   avec Et, Er normalises (norme 2 = 1), calcules
        uniquement sur les pixels valides de la pupille.

F = 1  -> la forme spatiale mesuree est identique au mode de Zernike pur.
F < 1  -> presence d'aberrations parasites / bruit / troncature de pupille.
"""

# %% ============================================================
# IMPORTS
# ============================================================
import numpy as np
import matplotlib.pyplot as plt
from maoppy.zernike import zernike, ansi2nm, ansi_name

plt.close('all')

# %% ============================================================
# PARAMETRES
# ============================================================

wavelength_nm = 632.8

# Mode injecte (indice ANSI/maoppy, PAS Noll standard) : 4 = defocus
target_j = 4

# Fichiers HASO : la reference "flat" (sans mode injecte) et les 3 niveaux
# de commande du mode 4 (defocus). Adapte les chemins si besoin.
base = r"C:\Users\ibrah\OneDrive\Bureau\stageLAM\Python_codes\Mesures_Stage_Lam\haso_lo"

file_flat = base + r"\HASO_1_nh.txt"

files_inj = {
    "m4":   base + r"\HASO_2_4_nh.txt",
    "m40":  base + r"\HASO_2_40_nh.txt",
    "m200": base + r"\HASO_2_200_nh.txt",
}

# %% ============================================================
# DETECTION DE PUPILLE SUR LA GRILLE HASO
#
# Les points hors pupille du HASO sont typiquement des NaN.
# On definit la pupille = zone valide, recentree sur son centroide,
# rayon = distance max des points valides au centroide.
# ============================================================

def pupil_geometry_from_mask(ref_map):
    mask_valid = ~np.isnan(ref_map)
    ys, xs = np.nonzero(mask_valid)
    cy, cx = ys.mean(), xs.mean()
    r_px = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2).max()
    Y, X = np.mgrid[0:ref_map.shape[0], 0:ref_map.shape[1]]
    Rn = np.sqrt(((X - cx) / r_px) ** 2 + ((Y - cy) / r_px) ** 2)
    pupil = mask_valid & (Rn <= 1.0)
    return pupil, cx, cy, r_px

# %% ============================================================
# GENERATION DU MODE DE ZERNIKE AVEC MAOPPY
#
# Meme appel que dans DMD_reconstruction_V2.py :
#   n, m = ansi2nm(j)
#   Z = zernike(n, m, npix, norm="noll", outside=0)
# maoppy genere un disque de npix x npix (pupille = disque inscrit,
# rayon = npix/2). On le place ici a la position/taille exactes de
# la pupille detectee sur la grille HASO (centre cx,cy, rayon r_px).
# ============================================================

def build_zernike_map(j, shape, cx, cy, r_px):
    n_j, m_j = ansi2nm(j)
    print(f"  Mode maoppy : j={j} (n={n_j}, m={m_j}) - {ansi_name(j)}")

    npix = int(round(2 * r_px))
    Z = np.nan_to_num(zernike(int(n_j), int(m_j), npix, norm="noll", outside=0))

    out = np.zeros(shape)
    y0 = int(round(cy - npix / 2))
    x0 = int(round(cx - npix / 2))
    ys, ye = max(y0, 0), min(y0 + npix, shape[0])
    xs, xe = max(x0, 0), min(x0 + npix, shape[1])
    zy0, zx0 = ys - y0, xs - x0
    out[ys:ye, xs:xe] = Z[zy0:zy0 + (ye - ys), zx0:zx0 + (xe - xs)]
    return out

# %% ============================================================
# CALCUL DE LA FIDELITE POUR UNE MESURE
# ============================================================

def compute_fidelity(file_inj, file_flat, target_j, wavelength_nm=632.8):
    data_inj  = np.loadtxt(file_inj,  skiprows=12)
    data_flat = np.loadtxt(file_flat, skiprows=12)

    diff_nm  = data_inj - data_flat
    diff_rad = diff_nm * (2 * np.pi / wavelength_nm)

    pupil, cx, cy, r_px = pupil_geometry_from_mask(diff_rad)

    rms_diff = np.sqrt(np.nanmean(diff_rad[pupil] ** 2))

    # Mode theorique (maoppy), mis a la meme RMS que la mesure (comparaison de forme)
    Z = build_zernike_map(target_j, diff_rad.shape, cx, cy, r_px)
    Z_rms_pupil = np.sqrt(np.mean(Z[pupil] ** 2))   # devrait etre ~1 deja (norm="noll")
    phi_cible = (rms_diff / Z_rms_pupil) * Z
    phi_cible = -phi_cible

    Et = np.exp(1j * phi_cible[pupil])
    Er = np.exp(1j * diff_rad[pupil])
    Et = Et / np.linalg.norm(Et)
    Er = Er / np.linalg.norm(Er)

    F = np.abs(np.vdot(Et, Er)) ** 2

    return {
        "F": F,
        "rms_diff_rad": rms_diff,
        "diff_rad": diff_rad,
        "phi_cible": phi_cible,
        "pupil": pupil,
    }

# %% ============================================================
# BOUCLE SUR LES 3 NIVEAUX DE COMMANDE (m4, m40, m200)
# ============================================================

results = {}
for label, fpath in files_inj.items():
    try:
        res = compute_fidelity(fpath, file_flat, target_j, wavelength_nm)
        results[label] = res
        print(f"{label:5s} : RMS mesure = {res['rms_diff_rad']:.4f} rad   "
              f"F = {res['F']:.5f}")
    except OSError as e:
        print(f"{label:5s} : fichier introuvable ({fpath}) -> {e}")

# %% ============================================================
# AFFICHAGE : phase mesuree vs phase cible, par niveau de commande
# ============================================================

n_ok = len(results)
if n_ok > 0:
    fig, axes = plt.subplots(2, n_ok, figsize=(5 * n_ok, 9))
    if n_ok == 1:
        axes = axes.reshape(2, 1)

    for col, (label, res) in enumerate(results.items()):
        vmax = np.nanmax(np.abs(res["diff_rad"][res["pupil"]]))

        im0 = axes[0, col].imshow(res["diff_rad"], cmap="jet", vmin=-vmax, vmax=vmax)
        axes[0, col].set_title(f"Mesure {label}\nRMS={res['rms_diff_rad']:.3f} rad")
        plt.colorbar(im0, ax=axes[0, col], label="Phase (rad)")

        im1 = axes[1, col].imshow(res["phi_cible"], cmap="jet", vmin=-vmax, vmax=vmax)
        axes[1, col].set_title(f"Zernike j={target_j} cible\nF={res['F']:.4f}")
        plt.colorbar(im1, ax=axes[1, col], label="Phase (rad)")

    fig.tight_layout()

# %% ============================================================
# FIDELITE EN FONCTION DU NIVEAU DE COMMANDE
# ============================================================

if n_ok > 0:
    labels = list(results.keys())
    Fs = [results[l]["F"] for l in labels]

    plt.figure("Fidelite vs commande")
    plt.plot(labels, Fs, "o-")
    plt.ylim(0, 1.05)
    plt.ylabel("Fidelite F")
    plt.xlabel("Niveau de commande")
    plt.title(f"Fidelite du mode Zernike j={target_j} (defocus) vs amplitude")
    plt.grid(True)

plt.show()