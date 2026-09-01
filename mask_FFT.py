# __________________Mask and FFT set up__________________


import numpy as np
import matplotlib.pyplot as plt

# On va mettre en place une image contennant un carré (basse fréquence) sur du fond hachuré (haute fréquence)
N = 512 
img = np.zeros((N,N))

img[N//3 : 2*N//3, N//3 : 2*N//3] = 1.0 
for i in range (0,N,4):
    img[:,i] += 0.5

"""# On affiche l'image source
plt.imshow(img, cmap='gray')
plt.title('Image source')
plt.axis('off')
plt.show()"""

#FFT plus centrage
F = np.fft.fft2(img)
F_shifted = np.fft.fftshift(F)
magnitude_spectrum = np.log(np.abs(F_shifted) + 1)  #+1 pour éviter log(0) et pour pouvoir l'affiher en échelle de gris

"""# On affiche le spectre de l'image
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Spectre de l\'image')
plt.axis('off')
plt.show()"""

# Création du masque circulaire
cy, cx = N//2, N//2  # Centre du spectre
Y,X = np.ogrid[:N, :N]
dist = np.sqrt((X-cx)**2 + (Y-cy)**2)
R = 10  # Rayon du masque
mask = (dist <= R).astype(float)

"""# Affichage du masque
plt.imshow(mask, cmap='gray')
plt.title('Masque circulaire')
plt.axis('off')
plt.show()"""

# Application du masque
F_shifted_filtered = F_shifted * mask
magnitude_filtered = np.log1p(np.abs(F_shifted_filtered))

"""plt.imshow(magnitude_filtered, cmap='gray')
plt.title('Image filtrée')
plt.axis('off')
plt.show()"""

# Inverse FFT pour obtenir l'image filtrée
F_back = np.fft.ifftshift(F_shifted_filtered)
img_filtered = np.real(np.fft.ifft2(F_back))

# Normalisation pour affichage
img_filtered = np.clip(img_filtered, 0, 1)

"""# Affichage de l'image filtrée
plt.imshow(img_filtered, cmap='gray')
plt.title('Image filtrée')
plt.axis('off')
plt.show()
"""

# onajoute le PSF de la tache d'Airy pour mieux visualiser le résultat 
# Calcul correct de la PSF
PSF = np.abs(np.fft.fftshift(np.fft.ifft2(mask)))**2  # fftshift pour centrer
PSF_norm = np.log1p(PSF / PSF.max()*1000)                   # log pour voir les anneaux
#img_filtered2 = img_filtered * PSF

# ── Visualisation ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 5, figsize=(16, 4))

axes[0].imshow(img,            cmap='gray', vmin=0, vmax=1)
axes[0].set_title('Image originale')

axes[1].imshow(np.log1p(np.abs(F_shifted)), cmap='inferno')
axes[1].set_title('|FFT| (log) — spectre centré')

axes[2].imshow(mask,           cmap='gray')
axes[2].set_title(f'Pupille (R={R})')

axes[3].imshow(img_filtered,   cmap='gray', vmin=0, vmax=1)
axes[3].set_title('Image filtrée (passe-bas)')

axes[4].imshow(PSF_norm, cmap='inferno')
axes[4].set_title('PSF (log, centrée)')

for ax in axes:
    ax.axis('off')
plt.tight_layout()
plt.show()