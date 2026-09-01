# %%
import numpy as np
from ALP4 import *
import time
from scipy.io import loadmat

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
#   -> si tu connais la zone réellement éclairée et qu'elle n'est
#      pas centrée, décale le motif avec np.roll ci-dessous
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

imgSeq = img.ravel()

# %% ------------------------------------------------------------
# ENVOI ET AFFICHAGE SUR LE DMD (identique à Test_python_dmd.py)
# ------------------------------------------------------------
DMD.SeqAlloc(nbImg=1, bitDepth=bitDepth)
DMD.SeqPut(imgData=imgSeq)
DMD.SetTiming(pictureTime=20000)

DMD.Run()

time.sleep(3600)

# %% ------------------------------------------------------------
# ARRÊT ET LIBÉRATION DU DMD
# ------------------------------------------------------------
DMD.Halt()
DMD.FreeSeq()
DMD.Free()