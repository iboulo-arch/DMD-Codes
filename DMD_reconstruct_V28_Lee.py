# %% ========================================================
#  MODULATION DE PHASE BINAIRE PAR HOLOGRAMME DE LEE
#  (pas de macro-pixels : la phase est codée pixel par pixel)
#
#  Portage Python de DMD_reconstruction_V2.m
#   -> la phase de Kolmogorov est remplacée par une phase de
#      Zernike (pupille elliptique, maoppy), comme dans
#      Test_python_dmdV6_lee_formElipt_rotModes.py
#
#  Références :
#   - W.H. Lee, Progress in Optics (1978)
#   - D.B. Conkey et al., Opt. Express (2012)
#   - https://www.wavefrontshaping.net/post/id/16
#
#  BUGS CORRIGÉS PAR RAPPORT AU SCRIPT MATLAB :
#   1) f_lee était construit comme 1*(1+cos(...)) donc dans
#      [0,2], mais binarisé au seuil 0.5 au lieu du milieu de
#      la plage (1). Le rapport cyclique du réseau n'était donc
#      pas de 50%, ce qui dégrade fortement l'isolement de
#      l'ordre +1 (fuite d'énergie vers l'ordre 0) et donc la
#      fidélité. Corrigé ici : f_lee = 0.5*(1+cos(...)) dans
#      [0,1], seuillé à 0.5 (conforme à la formule de Lee
#      rappelée en commentaire dans le .m).
#   2) La phase de Zernike est nulle par construction hors de
#      la pupille elliptique -> discontinuité franche au bord
#      de la pupille = contenu haute fréquence qui recouvre les
#      ordres +1/-1 et corrompt la reconstruction. Comme pour
#      le cas Kolmogorov, un filtre passe-bas de sécurité est
#      donc réappliqué après troncature par la pupille (à
#      désactiver via apply_safety_lowpass si le mode est bas
#      ordre et l'amplitude faible).
# ========================================================

# %% ------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from maoppy.zernike import zernike, ansi2nm, ansi_name

plt.close('all')

# %% ========================================================
# PARAMÈTRES
# ========================================================
N = 400                     # taille de la grille (pixels), pleine résolution

nu0 = 0.08                  # fréquence spatiale de la porteuse (cycles/pixel)
phi_cutoff = 0.127           # fréquence de coupure max pour phi (cycles/pixel)
                             # doit rester < nu0 - pinhole_radius
pinhole_radius = 0.022      # rayon du pinhole dans le plan de Fourier (cycles/pixel)

apply_safety_lowpass = True  # filtre passe-bas de sécurité sur phi (cf. bug #2 ci-dessus)

# --- pupille elliptique (mêmes conventions que le script DMD Python) ---
ellipse_e = 0
ellipse_theta_deg = -135    # orientation du grand axe (degrés, sens trigo)

# --- mode(s) de Zernike à injecter ---
use_combo = False           # False -> un seul mode (j_zernike/amp_zernike)
                             # True  -> combinaison pondérée (coeffs_zernike)

j_zernike = 200                # indice ANSI (4 = défocus)
amp_zernike = 1.0            # amplitude (rad)

coeffs_zernike = {           # utilisé seulement si use_combo=True
    3: -2.5,
    4: -0.3,
    5: 1.0,
}

# %% ========================================================
# GÉNÉRATION DE LA PHASE CIBLE (mode(s) de Zernike, pupille elliptique)
# ========================================================
print('Génération de la phase de Zernike...')


def build_zernike_phase(N, ellipse_e, ellipse_theta_deg,
                         j_zernike=None, amp_zernike=None, coeffs=None):
    if coeffs is not None:
        phi_out = np.zeros((N, N))
        for j, amp in coeffs.items():
            n_j, m_j = ansi2nm(j)
            Z_j = zernike(int(n_j), int(m_j), N, eccentricity=ellipse_e,
                          orientation=ellipse_theta_deg, norm="noll", outside=0)
            phi_out += amp * np.nan_to_num(Z_j)
            print(f"  + j={j} (n={n_j}, m={m_j}) {ansi_name(j)} : amp={amp}")
        return phi_out
    else:
        n_zer, m_zer = ansi2nm(j_zernike)
        print(f"Mode Zernike sélectionné : j={j_zernike} (n={n_zer}, m={m_zer}) - {ansi_name(j_zernike)}")
        return amp_zernike * np.nan_to_num(
            zernike(int(n_zer), int(m_zer), N, eccentricity=ellipse_e,
                    orientation=ellipse_theta_deg, norm="noll", outside=0))


phi_raw = build_zernike_phase(
    N, ellipse_e, ellipse_theta_deg,
    j_zernike=j_zernike, amp_zernike=amp_zernike,
    coeffs=coeffs_zernike if use_combo else None,
)

# --- masque de pupille elliptique -> amplitude cible ---
xx = np.arange(N)
X0, Y0 = np.meshgrid(xx, xx)
cx0 = cy0 = N / 2
a0 = N / 2
b0 = a0 * np.sqrt(1 - ellipse_e**2)
theta0 = np.deg2rad(ellipse_theta_deg)
xc = X0 - cx0
yc = Y0 - cy0
x_rot = xc * np.cos(theta0) + yc * np.sin(theta0)
y_rot = -xc * np.sin(theta0) + yc * np.cos(theta0)
pupil_mask = ((x_rot / a0) ** 2 + (y_rot / b0) ** 2) <= 1

amp_target = pupil_mask.astype(float)   # amplitude cible : 1 dans la pupille, 0 dehors
phi_raw = phi_raw * pupil_mask          # sécurité : phase strictement nulle hors pupille

# %% ========================================================
# FILTRE PASSE-BAS DE SÉCURITÉ (condition de bande du Lee hologram)
# ========================================================
fx = (np.arange(-N / 2, N / 2)) / N
FX, FY = np.meshgrid(fx, fx)
f_rad = np.sqrt(FX**2 + FY**2)
f_rad[f_rad == 0] = 1e-6

if apply_safety_lowpass:
    lowpass_mask = f_rad < phi_cutoff
    Phi_spec = np.fft.fftshift(np.fft.fft2(phi_raw)) * lowpass_mask
    phi = np.real(np.fft.ifft2(np.fft.ifftshift(Phi_spec)))
else:
    phi = phi_raw.copy()

# %% ========================================================
# CONSTRUCTION DE L'HOLOGRAMME DE LEE (pixel par pixel)
# ========================================================
print("Construction de l'hologramme de Lee...")

X, Y = np.meshgrid(np.arange(N), np.arange(N))
carrier = 2 * np.pi * nu0 * (X - Y)

f_lee = 0.5 * (1 + np.cos(carrier - phi))   # dans [0,1] (bug #1 corrigé)

# %% ========================================================
# BINARISATION -> motif envoyé au DMD
# ========================================================
DMD_pattern = (f_lee > 0.5).astype(float)

# %% ========================================================
# PLACEMENT SUR LE DMD PHYSIQUE (1024x768) -> nombre de pixels éclairés
#   -> purement informatif : la reconstruction/fidélité ci-dessous
#      continue de travailler sur la grille NxN (comme le .m).
#      Cette section replace juste le motif NxN au centre du vrai
#      chip DMD, comme fait dans Test_python_dmdV6_..., pour donner
#      un chiffre comparable à la taille physique du DMD.
# ========================================================
DMD_nSizeX, DMD_nSizeY = 1024, 768   # dimensions physiques du DMD (Vialux/ALP4)

assert N <= DMD_nSizeX and N <= DMD_nSizeY, \
    f"Le motif {N}x{N} est plus grand que le DMD ({DMD_nSizeX}x{DMD_nSizeY})"

pad_y = DMD_nSizeY - N
pad_x = DMD_nSizeX - N
pad_top, pad_bottom = pad_y // 2, pad_y - pad_y // 2
pad_left, pad_right = pad_x // 2, pad_x - pad_x // 2

DMD_pattern_full = np.pad(DMD_pattern, ((pad_top, pad_bottom), (pad_left, pad_right)),
                           mode='constant', constant_values=0)
pupil_mask_full = np.pad(pupil_mask, ((pad_top, pad_bottom), (pad_left, pad_right)),
                          mode='constant', constant_values=False)

n_pupil_px = int(pupil_mask_full.sum())            # pixels dans la pupille (zone utile)
n_on_px = int(DMD_pattern_full.sum())              # miroirs réellement ON (255) sur tout le chip
n_on_in_pupil = int((DMD_pattern_full.astype(bool) & pupil_mask_full).sum())

print()
print('=====================================')
print(f'Chip DMD : {DMD_nSizeX}x{DMD_nSizeY} = {DMD_nSizeX*DMD_nSizeY} px')
print(f'Pupille elliptique (zone utile)     : {n_pupil_px} px '
      f'({100*n_pupil_px/(DMD_nSizeX*DMD_nSizeY):.2f} % du chip)')
print(f'Miroirs ON (255) dans la pupille     : {n_on_in_pupil} px '
      f'({100*n_on_in_pupil/max(n_pupil_px,1):.2f} % de la pupille)')
print(f'Miroirs ON (255) sur tout le chip    : {n_on_px} px')
print(f'Miroirs OFF (hors pupille + hors motif ON) : {DMD_nSizeX*DMD_nSizeY - n_on_px} px')
print('=====================================')

# %% ========================================================
# SPECTRE DE FOURIER ET POSITION DU PINHOLE
# ========================================================
G = np.fft.fftshift(np.fft.fft2(DMD_pattern))

# l'ordre +1 est centré en (nu_x,nu_y) = (nu0,-nu0)
cx = -nu0
cy = nu0

mask = ((FX - cx) ** 2 + (FY - cy) ** 2) < pinhole_radius**2

# %% ========================================================
# FILTRAGE DU PINHOLE (simule le système 4f)
# ========================================================
print("Filtrage de l'ordre +1 de diffraction...")

Gfilt = G * mask

shift_row = int(round(cy * N))
shift_col = int(round(cx * N))
Gfilt_centered = np.roll(Gfilt, shift=(-shift_row, -shift_col), axis=(0, 1))

Efield = np.fft.ifft2(np.fft.ifftshift(Gfilt_centered))
phi_rec = np.angle(Efield)

# %% ========================================================
# FIDÉLITÉ DE MODULATION (pleine résolution, pixel par pixel)
# ========================================================
Et = amp_target * np.exp(1j * phi)     # champ cible = amplitude cible x phase cible
Er = Efield

Et_flat = Et.ravel() / np.linalg.norm(Et.ravel())
Er_flat = Er.ravel() / np.linalg.norm(Er.ravel())

Fidelity = np.abs(np.vdot(Et_flat, Er_flat)) ** 2

print()
print('=====================================')
print(f'nu0 = {nu0:.4f}, phi_cutoff = {phi_cutoff:.4f}, pinhole_radius = {pinhole_radius:.4f}')
print(f'Fidelity = {Fidelity:.5f}')
print(f'Error = {1 - Fidelity:.5f}')
print('=====================================')

# %% ========================================================
# AFFICHAGE
# ========================================================

amp_vmax = max(amp_target.max(), np.abs(Efield).max())
amp_cmap = 'jet'

fig, axs = plt.subplots(2, 4, figsize=(22, 9))

im = axs[0, 0].imshow(phi, cmap='jet')
axs[0, 0].set_title('Phase_target (Zernike 200)')
plt.colorbar(im, ax=axs[0, 0])

im = axs[0, 1].imshow(amp_target, cmap=amp_cmap, vmin=0, vmax=amp_vmax)
axs[0, 1].set_title('Amplitude_target ')
plt.colorbar(im, ax=axs[0, 1])

im = axs[0, 2].imshow(DMD_pattern, cmap='gray')
axs[0, 2].set_title(f'Hologramme binaire de Lee ({N}x{N}, $\\nu_0$={nu0:.3f})')
plt.colorbar(im, ax=axs[0, 2])

im = axs[0, 3].imshow(np.log(np.abs(G) + 1), extent=[fx[0], fx[-1], fx[0], fx[-1]],
                       origin='lower', cmap='viridis')
theta = np.linspace(0, 2 * np.pi, 100)
axs[0, 3].plot(cx + pinhole_radius * np.cos(theta), cy + pinhole_radius * np.sin(theta),
               'r-', linewidth=1.5)
axs[0, 3].plot(0, 0, 'y+', markersize=10, markeredgewidth=1.5)
axs[0, 3].plot(-cx - pinhole_radius * np.cos(theta), -cy - pinhole_radius * np.sin(theta),
               'c-', linewidth=1)
axs[0, 3].set_title('Spectre de Fourier + pinhole (rouge = ordre +1)')
axs[0, 3].set_xlabel(r'$\nu_x$ (cycles/pixel)')
axs[0, 3].set_ylabel(r'$\nu_y$ (cycles/pixel)')
plt.colorbar(im, ax=axs[0, 3])

im = axs[1, 0].imshow(phi_rec, cmap='jet')
axs[1, 0].set_title('Phase_out (après filtrage ordre +1)')
plt.colorbar(im, ax=axs[1, 0])

"""im = axs[1, 1].imshow(np.angle(np.exp(1j * (phi - phi_rec))), cmap='jet')
axs[1, 1].set_title('Erreur de phase (wrapped)')
plt.colorbar(im, ax=axs[1, 1])"""

im = axs[1, 2].imshow(np.abs(Efield), cmap=amp_cmap, vmin=0, vmax=amp_vmax)
axs[1, 2].set_title('Amplitude_out (après filtrage ordre +1)')
plt.colorbar(im, ax=axs[1, 2])

"""im = axs[1, 3].imshow(np.abs(Efield) - amp_target, cmap='RdBu_r')
axs[1, 3].set_title('Écart amplitude (reconstruite - cible)')
plt.colorbar(im, ax=axs[1, 3])"""

fig.suptitle(f'Reconstruction Hologramme de Lee (pixel par pixel) — '
             f'$\\nu_0$={nu0:.3f} — Fidelity={Fidelity:.4f}')
plt.tight_layout()
plt.show()

# %% ========================================================
# COMPARAISON DES PSD
# ========================================================
""""
PSD_target = np.abs(np.fft.fftshift(np.fft.fft2(phi))) ** 2
PSD_rec = np.abs(np.fft.fftshift(np.fft.fft2(phi_rec))) ** 2

plt.figure()
plt.loglog(np.mean(PSD_target[:, N // 2:], axis=0), linewidth=2, label='Cible')
plt.loglog(np.mean(PSD_rec[:, N // 2:], axis=0), linewidth=2, label='Reconstruite')
plt.grid(True, which='both')
plt.xlabel('Fréquence spatiale')
plt.ylabel('PSD')
plt.legend()
plt.title('Comparaison PSD (méthode hologramme de Lee, phase de Zernike)')
plt.show()
"""