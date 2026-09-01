# %%
import numpy as np
from ALP4 import *
import time

# %% ------------------------------------------------------------
# CONNEXION AU DMD
# ------------------------------------------------------------
DMD = ALP4(version='4.3', libDir="D:/ALP-4.3/ALP-4.3 API/")
DMD.Initialize()

# %% ------------------------------------------------------------
# PARAMÈTRES DE L'OUVERTURE CIRCULAIRE
#   -> à modifier pour déplacer/redimensionner l'ouverture
# ------------------------------------------------------------
cx = 0.98*DMD.nSizeX // 2   # centre de l'ouverture, axe x (pixels DMD)
cy = 1.06*DMD.nSizeY // 2   # centre de l'ouverture, axe y (pixels DMD)
radius = 10            # rayon de l'ouverture (pixels DMD)

# %% ------------------------------------------------------------
# CONSTRUCTION DU MASQUE : ON à l'int
# érieur du cercle, OFF ailleurs
# ------------------------------------------------------------
bitDepth = 1

x = np.arange(DMD.nSizeX)
y = np.arange(DMD.nSizeY)
X, Y = np.meshgrid(x, y)

inside_circle = ((X - cx)**2 + (Y - cy)**2) <= radius**2

outside_circle = ~inside_circle

imgOFF = np.zeros([DMD.nSizeY, DMD.nSizeX])
imgON  = np.ones([DMD.nSizeY, DMD.nSizeX]) * (2**8 - 1)

#img = np.where(inside_circle, imgON, imgOFF)
#imgSeq = img.ravel()

img = np.where(outside_circle, imgON, imgOFF)
imgSeq = img.ravel()

# %% ------------------------------------------------------------
# ENVOI ET AFFICHAGE SUR LE DMD (identique à Test_python_dmd.py)
# ------------------------------------------------------------
DMD.SeqAlloc(nbImg=1, bitDepth=bitDepth)
DMD.SeqPut(imgData=imgSeq)
DMD.SetTiming(pictureTime=20000)

DMD.Run()

time.sleep(120)

# %% ------------------------------------------------------------
# ARRÊT ET LIBÉRATION DU DMD
# ------------------------------------------------------------
DMD.Halt()
DMD.FreeSeq()
DMD.Free()