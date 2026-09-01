# %%
import h5py
import numpy as np
from ALP4 import *
import time
from scipy.io import loadmat
import scipy as sp
import matplotlib.pyplot as plt
from maoppy.zernike import zernike, ansi2nm, ansi_name

# %% ------------------------------------------------------------
# CONNEXION AU DMD
# ------------------------------------------------------------
DMD = ALP4(version='4.3', libDir="D:/ALP-4.3/ALP-4.3 API/")
DMD.Initialize()

# %% ------------------------------------------------------------
# CHARGEMENT DU MOTIF BINAIRE GÉNÉRÉ SOUS MATLAB
#   -> Le tableau `DMD` (Ndmd x Ndmd, valeurs 0/1) de
#      DMD_reconstruction_V2.m doit d'abord être exporté, ex :
#         save('DMD_pattern.mat','DMD')
#   -> Adapter le chemin ci-dessous si besoin
# ------------------------------------------------------------
mat_path = r"C:\Users\ibrah\OneDrive\Bureau\stageLAM\Matlab _codes\DMD_mask_focus_n4_0p8937_r0.100_20260722_103525.mat"

mat_data = loadmat(mat_path)
pattern = mat_data['dmd_mask']   #pour dmd _reconstruction_V2.m      
#pattern = mat_data['DMD']   # Ndmd x Ndmd, valeurs 0/1 

assert pattern.shape[0] == pattern.shape[1], "Le motif doit être carré"
assert set(np.unique(pattern)).issubset({0, 1}), "Le motif doit être binaire (0/1)"

Ndmd = pattern.shape[0]

# %% ------------------------------------------------------------
# POSITIONNEMENT DU MOTIF SUR LE DMD PHYSIQUE (1024x768)
#   -> le motif Ndmd x Ndmd (400x400) est centré sur toute la puce
#      via np.pad, le reste des miroirs restant OFF (0)
#  
# ------------------------------------------------------------
bitDepth = 1

assert Ndmd <= DMD.nSizeX and Ndmd <= DMD.nSizeY, \
    f"Le motif {Ndmd}x{Ndmd} est plus grand que le DMD ({DMD.nSizeX}x{DMD.nSizeY})"

pad_y = DMD.nSizeY - Ndmd
pad_x = DMD.nSizeX - Ndmd

pad_top    = pad_y // 2
pad_bottom = pad_y - pad_top
pad_left   = pad_x // 2
pad_right  = pad_x - pad_left

img = np.pad(
    pattern.astype(np.uint8) * (2**8 - 1),
    ((pad_top, pad_bottom), (pad_left, pad_right)),
    mode='constant',
    constant_values=0
)

# Optionnel : si le motif doit être décalé par rapport au centre
# géométrique de la puce (ex: zone éclairée décentrée), décommenter :
# shift_y, shift_x = 0, 0   # décalage en pixels DMD
# img = np.roll(img, shift=(shift_y, shift_x), axis=(0, 1))


L = np.arange(0,DMD.nSizeX)
l = np.arange(0,DMD.nSizeY)

x,y = np.meshgrid(L,l)

amp_focus = 0.001
phi = amp_focus * ((x-DMD.nSizeX//2)**2+(y-DMD.nSizeY//2)**2) #phi_focus

# ----------------------Envoie de modes KL----------------------
modal_basis_file_path = r"C:\Users\ibrah\OneDrive\Bureau\stageLAM\Python_codes\utc_2026-04-28_09-37-25_KL_modal_basis.h5"
with h5py.File(
    modal_basis_file_path,
    mode="r",
) as f:
    modal_basis = f["modal_basis"][...]
mode_number = 3
amp_mode = 0
phi_KL_raw = amp_mode *modal_basis[mode_number, :, :]   # 2D : (Ny, Nx)


Ndmd, Ndmd = phi_KL_raw.shape

pad_y = DMD.nSizeY - Ndmd
pad_x = DMD.nSizeX - Ndmd

pad_top    = pad_y // 2
pad_bottom = pad_y - pad_top
pad_left   = pad_x // 2
pad_right  = pad_x - pad_left

phi_KL = np.pad(
    phi_KL_raw,
    ((pad_top, pad_bottom), (pad_left, pad_right)),
    mode='constant',
    constant_values=0
)


#zoom_y = DMD.nSizeY / ny   # 768 / ny
#zoom_x = DMD.nSizeX / nx   # 1024 / nx
#phi_KL = sp.ndimage.zoom(phi_KL_raw, (zoom_y, zoom_x))

#assert phi_KL.shape == x.shape, f"Mismatch: {phi_KL.shape} vs {x.shape}"



# ----------------------Envoie de modes de Zernike (maoppy)----------------------
# Même logique que pour les modes KL : génération sur la pupille (npix_zernike x npix_zernike),
# puis padding à la taille complète du DMD -> phi_zernike prêt à être injecté dans le cos().

j_zernike = 10         # indice ANSI du mode (0=piston, 1/2=tilt, 4=defocus, 5=astig.+, 7=coma verticale, ...)
amp_zernike = -0.1     # amplitude du mode (mêmes unités que amp_mode / amp_focus, en rad)
npix_zernike = 600 #phi_KL_raw.shape[0]  # même résolution de pupille que la base KL, pour rester cohérent
ellipse_theta_deg = 45              # orientation du grand axe (degrés, sens trigo)
ellipse_e = 0.6

n_zer, m_zer = ansi2nm(j_zernike)
phi_zernike_raw = amp_zernike * np.nan_to_num(
    zernike(int(n_zer), int(m_zer), npix_zernike, eccentricity=ellipse_e, orientation=ellipse_theta_deg, norm="noll", outside=0)
)
print(f"Mode Zernike sélectionné: j={j_zernike} (n={n_zer}, m={m_zer}) - {ansi_name(j_zernike)}")


pad_y_zer = DMD.nSizeY - phi_zernike_raw.shape[0]
pad_x_zer = DMD.nSizeX - phi_zernike_raw.shape[1]

pad_top_zer    = pad_y_zer // 2
pad_bottom_zer = pad_y_zer - pad_top_zer
pad_left_zer   = pad_x_zer // 2
pad_right_zer  = pad_x_zer - pad_left_zer

phi_zernike = np.pad(
    phi_zernike_raw,
    ((pad_top_zer, pad_bottom_zer), (pad_left_zer, pad_right_zer)),
    mode='constant',
    constant_values=0
)


# ----------------------Combinaison de modes de Zernike (maoppy)----------------------
# Même principe que ci-dessus, mais on somme plusieurs modes pondérés par un coefficient
# chacun -> phi_zernike_combo prêt à être injecté dans le cos(), à la place de phi_zernike.

coeffs_zernike = {1: 0, 2: 0, 3: -2.5, 4: -0.3, 5: 1, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0}  # {indice ANSI: amplitude (rad)}; indice ANSI du mode (0=piston, 1/2=tilt, 4=defocus, 5=astig.+, 7=coma verticale, ...)


phi_zernike_combo_raw = np.zeros((npix_zernike, npix_zernike))
for j, amp in coeffs_zernike.items():
    n_j, m_j = ansi2nm(j)
    Z_j = zernike(int(n_j), int(m_j), npix_zernike, eccentricity=ellipse_e, orientation=ellipse_theta_deg,norm="noll", outside=0)
    phi_zernike_combo_raw += amp * np.nan_to_num(Z_j)
    print(f"  + j={j} (n={n_j}, m={m_j}) {ansi_name(j)} : amp={amp}")

phi_zernike_combo = np.pad(
    phi_zernike_combo_raw,
    ((pad_top_zer, pad_bottom_zer), (pad_left_zer, pad_right_zer)),
    mode='constant',
    constant_values=0
)

# %% ------------------------------------------------------------
# MASQUE ELLIPTIQUE : la zone hors de l'ellipse reste "plate"
# (miroirs OFF -> pas de faisceau diffracté dans la direction d'étude)
# ------------------------------------------------------------

# --- Paramètres de l'ellipse (en pixels DMD) ---
ellipse_cx = DMD.nSizeX // 2      # centre X (pixels)
ellipse_cy = DMD.nSizeY // 2      # centre Y (pixels)
ellipse_a  = npix_zernike/2                  # demi-grand axe (pixels)

# Choisir UNE des deux options ci-dessous pour fixer le petit axe :

# Option 1 : donner directement le demi-petit axe b
#ellipse_b = 200                   # demi-petit axe (pixels)

# Option 2 : donner l'excentricité e (0 = cercle, ->1 = ellipse très aplatie)
# et calculer b à partir de a et e (décommenter pour utiliser) :

ellipse_b = ellipse_a * np.sqrt(1 - ellipse_e**2)

def make_elliptical_mask(x, y, cx, cy, a, b, theta_deg):
    """
    Retourne un masque booléen (True = intérieur de l'ellipse).
    x, y : meshgrid de coordonnées (mêmes dimensions que le pattern)
    cx, cy : centre de l'ellipse (pixels)
    a, b : demi-grand axe et demi-petit axe (pixels)
    theta_deg : rotation de l'ellipse (degrés, sens trigo)
    """
    theta = np.deg2rad(theta_deg)
    xc = x - cx
    yc = y - cy
    x_rot =  xc * np.cos(theta) + yc * np.sin(theta)
    y_rot = -xc * np.sin(theta) + yc * np.cos(theta)
    return (x_rot / a) ** 2 + (y_rot / b) ** 2 <= 1

ellipse_mask = make_elliptical_mask(
    x, y,
    ellipse_cx, ellipse_cy,
    ellipse_a, ellipse_b,
    ellipse_theta_deg
)

# Excentricité effective (utile pour vérification / logs)
ellipse_e_eff = np.sqrt(1 - (ellipse_b / ellipse_a) ** 2) if ellipse_a >= ellipse_b else np.nan
print(f"Ellipse: a={ellipse_a}, b={ellipse_b}, e={ellipse_e_eff:.3f}, "
      f"centre=({ellipse_cx},{ellipse_cy}), theta={ellipse_theta_deg}°")



# %% ------------------------------------------------------------
# GÉNÉRATION DU MOTIF ET APPLICATION DU MASQUE
# ------------------------------------------------------------
freq = 0.8

f = 1 + np.cos(freq * (x + y) - (phi_zernike_combo+phi_zernike))  # f varie entre 0 et 2, seuil à 1 -> motif binaire, phi_zernike if Zernik mode
#                                              phi_zernike_combo pour la combinaison

sp.ndimage.zoom(f, (DMD.nSizeY / f.shape[0], DMD.nSizeX / f.shape[1]), order=1, mode='nearest', prefilter=False)

f[f > 1/2] = 255  # 255 joue le rôle de 1 (DMD en 8 bits, 0-255)
f[f <= 1/2] = 0

# --- Application du masque elliptique : hors ellipse -> plat (miroirs OFF) ---
f[~ellipse_mask] = 0
#np.save(r"C:\Users\ibrah\OneDrive\Bureau\stageLAM\Python_codes\f_200.npy", f)
plt.figure()
plt.imshow(f)
plt.title("Motif final masqué par l'ellipse")
plt.show()

# --- Motif effectivement envoyé au DMD (remplace l'ancien "tout à 255") ---
imgSeq = f.astype(np.uint8).ravel()

# %% ------------------------------------------------------------
# ENVOI ET AFFICHAGE SUR LE DMD (identique à Test_python_dmd.py)
# ------------------------------------------------------------
DMD.SeqAlloc(nbImg=1, bitDepth=bitDepth)
DMD.SeqPut(imgData=imgSeq)
DMD.SetTiming(pictureTime=10000000)

DMD.Run()

time.sleep(3600)

# %% ------------------------------------------------------------
# ARRÊT ET LIBÉRATION DU DMD
# ------------------------------------------------------------
DMD.Halt()
DMD.FreeSeq()
DMD.Free()