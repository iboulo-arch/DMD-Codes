import numpy as np
import matplotlib.pyplot as plt

N = 512

# ── 1. Masque circulaire = pupille du télescope ──────────────────
cy, cx = N // 2, N // 2
Y, X = np.ogrid[:N, :N]
dist = np.sqrt((X - cx)**2 + (Y - cy)**2)

R = 50
masque_pupille = (dist <= R).astype(complex)

# ── 2. Carte de lumière incidente (onde plane)
carte_lumiere = np.ones((N, N), dtype=complex)

# ── 3. Champ à l'entrée = carte de lumière × masque
champ_entree = carte_lumiere * masque_pupille

# ── 4. FFT → plan focal
champ_focal = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(champ_entree)))

# ── 5. Intensité
intensite = np.abs(champ_focal)**2

# ── 6. Figure avec 3 panneaux ────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

axes[0].imshow(np.abs(carte_lumiere), cmap='gray', origin='lower')
axes[0].set_title('Carte de lumière incidente')

axes[1].imshow(np.abs(champ_entree), cmap='gray', origin='lower')
axes[1].set_title('Champ d\'entrée |amplitude|')

axes[2].imshow(np.log1p(intensite), cmap='gray', origin='lower')
axes[2].set_title('Tache d\'Airy — plan focal')

for ax in axes:
    ax.axis('off')

plt.suptitle('Diffraction de Fraunhofer — ouverture circulaire', fontsize=13)
plt.tight_layout()
plt.show()