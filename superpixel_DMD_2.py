# -*- coding: utf-8 -*-
"""
super_pixel_DMD.py  —  Goorden 2014, vrai DMD +/-12 deg
Traduction Python de super_pixel_DMD.m (compatible cellules Spyder/VS Code)

Modele physique :
    Miroirs +/-theta=+/-12 deg, eclairage Littrow a 2*theta=24 deg.
    ON  (+12 deg) -> 0 deg  -> tache speculaire ON = origine plan focal.
    OFF (-12 deg) -> -48 deg -> ~148 mm off-axis (filtre).

Modele de simulation :
    ON  pixel -> amplitude 1  (phase 0 dans le plan miroir)
    OFF pixel -> porteur proxy a (f_off, f_off) = (0.40, 0.40) cy/miroir
                 Pour Ndmd=400 : contribution OFF a l'ouverture = EXACTEMENT 0

    Ouverture a (f0x, f0y) = (-1/n^2, +1/n) cy/miroir [position Goorden].

LUT :
    Coefficient DFT direct a (f0x, f0y) :
        Estate(s) = somme_{ON (ir,ic)} exp(-i*2*pi*(f0x*ic + f0y*ir))
    OFF ignore (contribution nulle en propagation complete).

Extraction champ reconstruit :
    1. IFFT(FFT(Device) x H_ouverture)
    2. Demodulation : x exp(-i*2*pi*(f0x*IC + f0y*IR))
    3. Moyenne sur chaque bloc n x n
    4. Correction porteur colonne : x exp(-i*2*pi*ix/n)
"""

# %% ============================================================
# IMPORTS
# ============================================================
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
from datetime import datetime
from maoppy.zernike import zernike, ansi2nm, ansi_name
import os

plt.close('all')

# %% ============================================================
# PARAMETRES
# ============================================================

Nsp  = 100
n    = 4
Ndmd = Nsp * n          # = 400

lam  = 633e-9           # lambda (m)
d    = 13.68e-6         # pas miroir DMD (m)
theta = 12 * np.pi / 180

f_L2 = 300e-3

# 'turbulence' | 'LG' | 'tiptilt' | 'focus' | 'flat' | 'zernike'
target_mode = 'zernike'

# --- Aberrations additionnelles (s'ajoutent a la phase de base) ---
add_tiptilt = False     # True = ajouter tip/tilt
tip_rad     = 2.0       # amplitude tip  [rad peak-to-edge, axe x]
tilt_rad    = 3.1       # amplitude tilt [rad peak-to-edge, axe y]

add_focus   = False     # True = ajouter focus (defocalisation)
focus_rad   = 1.0       # amplitude focus [rad peak-to-edge]

# --- Modes de Zernike (utilise seulement si target_mode = 'zernike') ---
# Indices maoppy (ansi2nm), PAS l'indexation Noll standard : au-dela des
# premiers modes, maoppy.ansi2nm() ne suit pas la meme convention que la
# formule Noll classique (ex: j=40 -> Noll donne m=4, maoppy donne m=0).
# Utiliser maoppy directement evite le mismatch (modes en "lobes" au lieu
# d'anneaux concentriques). Ex : [4] defocus, [5,6] astigmatisme...
# Amplitude en rad RMS sur la pupille (rho<=1).
zernike_noll_indices = [4]        # ex : [4] defocus, [5,6] astigmatisme...
zernike_coeffs_rad   = [1.0]      # amplitude RMS associee a chaque mode

# --- Amplitude cible [0 -> 1] ---
#   Pour cibles phase-pure : garder 1.0 (les points LUT du bord donnent
#   la meilleure couverture angulaire, la fidelite est invariante apres norm.)
#   Utile seulement pour cibles amplitude+phase (ex. LG) : reduire < 1.
amp_target = 1.0

# %% ============================================================
# GEOMETRIE PHYSIQUE
# ============================================================

theta_illum       = 2 * theta
theta_OFF_reflect = 2 * (-theta) - theta_illum          # -48 deg
x_OFF_phys        = f_L2 * np.sin(theta_OFF_reflect)    # ~ -148 mm

# %% ============================================================
# OUVERTURE GOORDEN
# ============================================================

# Frequences normalisees [cy/miroir]
f0x_norm = -1 / n**2      # = -0.0625
f0y_norm = +1 / n         # = +0.25

# Position physique dans le plan focal [m]
x_ap = f0x_norm * lam * f_L2 / d
y_ap = f0y_norm * lam * f_L2 / d

# Rayon [cy/miroir] -- critere Nyquist : r < 1/(2n) = 0.125
r_norm = 0.10
r_phys = r_norm * lam * f_L2 / d

print(f'=== DMD +/-{theta*180/np.pi:.0f} deg  (Littrow {theta_illum*180/np.pi:.0f} deg) ===')
print('Tache ON         : (0, 0) mm')
print(f'Tache OFF reelle : ({x_OFF_phys*1e3:.1f}, 0) mm  [filtree]')
print(f'\nPinhole sur le banc (plan focal L2, f={f_L2*1e3:.0f} mm) :')
print(f'  Centre : ({x_ap*1e3:.3f}, {y_ap*1e3:.3f}) mm')
print(f'  Rayon  : {r_phys*1e3:.3f} mm  (diam. {2*r_phys*1e3:.3f} mm)')
ok_str = 'OK' if r_norm < 1/(2*n) else 'FAIL'
print(f'  r norm : {r_norm:.4f}  < 1/(2n)={1/(2*n):.4f}  -> {ok_str}\n')

# %% ============================================================
# PORTEUR MIROIR OFF  (proxy simulation)
#
#   f_off = 0.40 cy/miroir
#   (f_off - f0x) x 400 = (0.40+0.0625)x400 = 185  <- entier
#   (f_off - f0y) x 400 = (0.40-0.25)x400   =  60  <- entier
#   -> contribution OFF a l'ouverture = 0 EXACTEMENT pour Ndmd=400
# ============================================================

f_off = 0.40

IC_g, IR_g = np.meshgrid(np.arange(Ndmd), np.arange(Ndmd))
phi_off_g  = 2 * np.pi * (f_off * IC_g + f_off * IR_g)     # porteur global OFF

ic_sp, ir_sp = np.meshgrid(np.arange(n), np.arange(n))
phi_off_sp   = 2 * np.pi * (f_off * ic_sp + f_off * ir_sp)  # porteur local OFF

# Noyau DFT a (f0x, f0y)
demod_kernel = np.exp(-1j * 2 * np.pi * (f0x_norm * ic_sp + f0y_norm * ir_sp))  # n x n

# %% ============================================================
# GRILLE Zernike : la pupille (Rn<=1) sur Xn,Yn in [-1,1] avec Nsp
# points correspond exactement au disque inscrit que genere
# maoppy.zernike(n, m, Nsp, ...) -> pas de recentrage necessaire.
# ============================================================

# %% ============================================================
# CIBLE
# ============================================================

# Grille normalisee commune [-1, +1]
Xn, Yn = np.meshgrid(np.linspace(-1, 1, Nsp), np.linspace(-1, 1, Nsp))
Rn = np.sqrt(Xn**2 + Yn**2)
Thn = np.arctan2(Yn, Xn)

Etarget = None  # sera defini plus bas (sauf cas 'lg' qui le fixe directement)

tm = target_mode.lower()

if tm == 'turbulence':
    print('Cible : turbulence de Kolmogorov')
    fx_t = (np.arange(Nsp) - Nsp / 2) / Nsp
    FXt, FYt = np.meshgrid(fx_t, fx_t)
    ft = np.sqrt(FXt**2 + FYt**2)
    ft[ft == 0] = 1e-6
    PSD = ft ** (-11 / 3)
    cn = np.random.randn(Nsp, Nsp) + 1j * np.random.randn(Nsp, Nsp)
    phi = np.real(np.fft.ifft2(np.fft.ifftshift(cn * np.sqrt(PSD))))
    phi = phi / np.std(phi) * np.pi

elif tm == 'tiptilt':
    print('Cible : tip/tilt pur')
    phi = np.zeros((Nsp, Nsp))

elif tm == 'focus':
    print('Cible : focus pur')
    phi = np.zeros((Nsp, Nsp))

elif tm == 'flat':
    print('Cible : phase nulle (champ uniforme)')
    phi = np.zeros((Nsp, Nsp))

elif tm == 'lg':
    print('Cible : LG_1^0')
    PHI = np.arctan2(Yn, Xn)
    A = Rn * np.exp(-Rn**2)
    Etarget = (A / np.max(A)) * np.exp(1j * PHI)
    phi = np.angle(Etarget)

elif tm == 'zernike':
    print('Cible : combinaison de modes de Zernike (maoppy)')
    pupil = (Rn <= 1)
    phi = np.zeros((Nsp, Nsp))
    for j_idx, coeff in zip(zernike_noll_indices, zernike_coeffs_rad):
        n_z, m_z = ansi2nm(j_idx)
        mode = np.nan_to_num(zernike(int(n_z), int(m_z), Nsp, norm="noll", outside=0))
        phi += coeff * mode
        print(f'  + Zernike maoppy j={j_idx} (n={n_z}, m={m_z}) - {ansi_name(j_idx)}  '
              f'amplitude = {coeff:.3f} rad RMS')

else:
    raise ValueError(
        f"target_mode inconnu : '{target_mode}'\n"
        "Modes valides : turbulence | tiptilt | focus | flat | LG | zernike"
    )

# --- Aberrations additionnelles superposees a phi ---
phi_aberr = np.zeros((Nsp, Nsp))

if add_tiptilt:
    phi_aberr = phi_aberr + tip_rad * Xn + tilt_rad * Yn
    print(f'  + Tip  = {tip_rad:.2f} rad   Tilt = {tilt_rad:.2f} rad')

if add_focus:
    phi_aberr = phi_aberr + focus_rad * (1 - Rn**2)
    print(f'  + Focus = {focus_rad:.2f} rad (pic au centre)')

phi = phi + phi_aberr

# Champ cible complexe
if Etarget is None:
    Etarget = np.exp(1j * phi)

# %% ============================================================
# BUILD LUT
#
# Estate(s) = somme_{ON (ir,ic)} exp(-i*2*pi*(f0x*ic + f0y*ir))
#
# Calcul vectorise : produit matriciel du masque ON (tous les etats)
# avec le noyau DFT a la frequence ouverture. OFF ignore (contribution
# nulle a l'ouverture en propagation complete pour Ndmd=400, f_off=0.40).
# ============================================================

print(f'LUT ({2**(n**2)} etats) ...')

Nstates = 2 ** (n**2)

# Bits[s, k] = bit k (0-indexe, LSB=bit0) de l'etat s
bit_idx = np.arange(n**2)
Bits = ((np.arange(Nstates)[:, None] >> bit_idx[None, :]) & 1).astype(np.uint8)

# demod_kernel apalti dans le meme ordre que le "reshape colonne-major"
# MATLAB : block(r,c) = bits(c*n + r)  ->  kernel_flat[c*n+r] = demod_kernel[r,c]
kernel_flat = demod_kernel.flatten(order='F')

Estate = Bits.astype(np.complex128) @ kernel_flat
Estate = Estate / np.max(np.abs(Estate))

# --- Amplitude optimale : met le cercle cible dans la zone dense de la LUT ---
if isinstance(amp_target, str) and amp_target.lower() == 'auto':
    A_opt = np.median(np.abs(Estate))
else:
    A_opt = amp_target
print(f'Amplitude cible  : {A_opt:.4f}  (mediane LUT = {np.median(np.abs(Estate)):.4f})')

# Rescaler Etarget a l'amplitude optimale (phase inchangee)
Etarget = A_opt * np.exp(1j * np.angle(Etarget))

plt.figure('LUT')
plt.plot(Estate.real, Estate.imag, '.', markersize=2)
th_lut = np.linspace(0, 2*np.pi, 300)
plt.plot(A_opt*np.cos(th_lut), A_opt*np.sin(th_lut), 'r-', linewidth=1.5)
plt.axis('equal'); plt.grid(True)
plt.xlabel('Re(E)'); plt.ylabel('Im(E)')
plt.legend(['LUT', f'Cible A={A_opt:.3f}'], loc='best')
plt.title(f'DMD LUT — n={n}, r={r_norm:.3f} cy/miroir')

# %% ============================================================
# BUILD DMD DEVICE + RECONSTRUCTION
# ============================================================

print('Device et reconstruction ...')

ComplexDevice = np.exp(1j * phi_off_g)           # tout initialise a OFF
BinaryDevice  = np.zeros((Ndmd, Ndmd))
Ereconstructed = np.zeros((Nsp, Nsp), dtype=complex)

Etarget_flat = Etarget.reshape(-1)
diffs_base = np.abs(Estate)  # reutilise juste pour preallouer si besoin

for iy in range(Nsp):
    for ix in range(Nsp):
        target = Etarget[iy, ix]
        idx = np.argmin(np.abs(Estate - target))
        Ereconstructed[iy, ix] = Estate[idx]

        bits = Bits[idx, :]
        block = bits.reshape(n, n, order='F').astype(bool)

        ys, xs = iy * n, ix * n

        tile = np.exp(1j * phi_off_sp)           # OFF
        tile[block] = 1                          # ON = amplitude 1

        ComplexDevice[ys:ys+n, xs:xs+n] = tile
        BinaryDevice[ys:ys+n, xs:xs+n]  = block.astype(float)

# %% ============================================================
# PROPAGATION FOURIER COMPLETE
# ============================================================

print('Propagation Fourier ...')

# --- FFT sans padding -> reconstruction du champ ---
Frecon = np.fft.fftshift(np.fft.fft2(ComplexDevice))

fx_r = (np.arange(Ndmd) - Ndmd/2) / Ndmd
FXr, FYr = np.meshgrid(fx_r, fx_r)
Hr = (((FXr - f0x_norm)**2 + (FYr - f0y_norm)**2) < r_norm**2).astype(float)
Ffilt = Frecon * Hr
ReconFull = np.fft.ifft2(np.fft.ifftshift(Ffilt))

# --- FFT zero-paddee -> affichage plan focal haute resolution ---
pad_factor = 8            # 8x -> ~8 pixels par resolution diffraction
Npad = pad_factor * Ndmd

DevicePad = np.zeros((Npad, Npad), dtype=complex)
off_p = (Npad - Ndmd) // 2
DevicePad[off_p:off_p+Ndmd, off_p:off_p+Ndmd] = ComplexDevice

Ffull = np.fft.fftshift(np.fft.fft2(DevicePad))

# Axes physiques plan focal [mm]
dx_mm = lam * f_L2 / (Npad * d) * 1e3
x_ax = (np.arange(Npad) - Npad/2) * dx_mm

# Filtre d'ouverture dans la grille paddee
fx_pad = (np.arange(Npad) - Npad/2) / Npad     # [cy/miroir]
FXp, FYp = np.meshgrid(fx_pad, fx_pad)
Hfull = (((FXp - f0x_norm)**2 + (FYp - f0y_norm)**2) < r_norm**2).astype(float)
Ffilt_disp = Ffull * Hfull

# %% ============================================================
# AFFICHAGE PLAN FOCAL
# ============================================================

Imax = np.max(np.abs(Ffull))**2
Ilog_no = 10*np.log10(np.abs(Ffull)**2      / Imax + 1e-8)
Ilog_pi = 10*np.log10(np.abs(Ffilt_disp)**2 / Imax + 1e-8)

th_c = np.linspace(0, 2*np.pi, 300)
xc_mm = x_ap*1e3 + r_phys*1e3*np.cos(th_c)
yc_mm = y_ap*1e3 + r_phys*1e3*np.sin(th_c)

x_off_sim_mm = f_off * lam * f_L2 / d * 1e3     # position proxy OFF

# --- SANS pinhole ---
plt.figure('Plan focal — SANS pinhole', figsize=(12, 5.6))
plt.imshow(Ilog_no, extent=[x_ax[0], x_ax[-1], x_ax[0], x_ax[-1]],
           origin='lower', cmap='hot', vmin=-60, vmax=0)
plt.colorbar(label='dB'); plt.axis('image')
plt.xlabel('x [mm]'); plt.ylabel('y [mm]')
plt.title(f'Plan focal — SANS pinhole   (lambda={round(lam*1e9)}nm, '
          f'f={f_L2*1e3:.0f}mm, d={d*1e6:.2f}um)')
plt.plot(0, 0, 'g+', markersize=16, markeredgewidth=2.5)
plt.text(0.10, -0.22, 'Tache ON (0,0)', color='g', fontweight='bold', fontsize=9)
plt.plot(xc_mm, yc_mm, 'c-', linewidth=2.5)
plt.plot(x_ap*1e3, y_ap*1e3, 'c+', markersize=12, markeredgewidth=2.5)
plt.text(x_ap*1e3+0.15, y_ap*1e3+0.12,
          f'Pinhole Goorden\n({x_ap*1e3:.2f}, {y_ap*1e3:.2f}) mm\nr = {r_phys*1e3:.2f} mm',
          color='c', fontsize=9, backgroundcolor='k')
plt.plot(x_off_sim_mm, x_off_sim_mm, 'r^', markersize=9, markeredgewidth=1.5)
plt.text(x_off_sim_mm+0.12, x_off_sim_mm,
          f'OFF proxy\n[reel: {abs(x_OFF_phys*1e3):.0f}mm]', color='r', fontsize=8)

# --- AVEC pinhole ---
plt.figure('Plan focal — AVEC pinhole', figsize=(12, 5.6))
plt.imshow(Ilog_pi, extent=[x_ax[0], x_ax[-1], x_ax[0], x_ax[-1]],
           origin='lower', cmap='hot', vmin=-60, vmax=0)
plt.colorbar(label='dB'); plt.axis('image')
plt.xlabel('x [mm]'); plt.ylabel('y [mm]')
plt.title(f'Plan focal — AVEC pinhole   (r = {r_norm:.3f} cy/miroir = {r_phys*1e3:.3f} mm)')
plt.plot(xc_mm, yc_mm, 'c-', linewidth=2.5)
plt.plot(x_ap*1e3, y_ap*1e3, 'c+', markersize=12, markeredgewidth=2.5)
plt.text(x_ap*1e3+0.15, y_ap*1e3+0.12,
          f'Pinhole\n({x_ap*1e3:.2f}, {y_ap*1e3:.2f}) mm', color='c', fontsize=9)
plt.plot(0, 0, 'g+', markersize=12, markeredgewidth=2)

# --- Zoom ouverture ---
zoom_h = r_phys*1e3 * 8

fig, axs = plt.subplots(1, 2, num='Zoom : region pinhole', figsize=(10, 4.5))
im0 = axs[0].imshow(Ilog_no, extent=[x_ax[0], x_ax[-1], x_ax[0], x_ax[-1]],
                     origin='lower', cmap='hot', vmin=-60, vmax=0)
fig.colorbar(im0, ax=axs[0]); axs[0].axis('image')
axs[0].set_xlim(x_ap*1e3 - zoom_h, x_ap*1e3 + zoom_h)
axs[0].set_ylim(y_ap*1e3 - zoom_h, y_ap*1e3 + zoom_h)
axs[0].set_xlabel('x [mm]'); axs[0].set_ylabel('y [mm]'); axs[0].set_title('Zoom — sans pinhole')
axs[0].plot(xc_mm, yc_mm, 'c-', linewidth=2)

im1 = axs[1].imshow(Ilog_pi, extent=[x_ax[0], x_ax[-1], x_ax[0], x_ax[-1]],
                     origin='lower', cmap='hot', vmin=-60, vmax=0)
fig.colorbar(im1, ax=axs[1]); axs[1].axis('image')
axs[1].set_xlim(x_ap*1e3 - zoom_h, x_ap*1e3 + zoom_h)
axs[1].set_ylim(y_ap*1e3 - zoom_h, y_ap*1e3 + zoom_h)
axs[1].set_xlabel('x [mm]'); axs[1].set_ylabel('y [mm]'); axs[1].set_title('Zoom — avec pinhole')
axs[1].plot(xc_mm, yc_mm, 'c-', linewidth=2)
fig.tight_layout()

# %% ============================================================
# EXTRACTION DU CHAMP RECONSTRUIT
#
# Apres filtrage off-axis a (f0x,f0y), ReconFull porte le porteur
# spatial exp(+i*2*pi*(f0x*IC + f0y*IR)).
# 1. Demodulation -> champ lentement variable
# 2. Moyenne par bloc n x n
# 3. Correction porteur de colonne x exp(-i*2*pi*ix/n)
# ============================================================

IC_d, IR_d = np.meshgrid(np.arange(Ndmd), np.arange(Ndmd))
ReconDemod = ReconFull * np.exp(-1j*2*np.pi*(f0x_norm*IC_d + f0y_norm*IR_d))

Efinal = np.zeros((Nsp, Nsp), dtype=complex)
for iy in range(Nsp):
    for ix in range(Nsp):
        ys, xs = iy*n, ix*n
        blk = ReconDemod[ys:ys+n, xs:xs+n]
        Efinal[iy, ix] = np.mean(blk)

# Correction porteur de colonne :
# Efinal(iy,ix) ~ exp(+i*2*pi*ix/n) x Estate(best_s)  -> multiplier par exp(-i*2*pi*ix/n)
IX, _ = np.meshgrid(np.arange(Nsp), np.arange(Nsp))
Efinal = Efinal * np.exp(-1j*2*np.pi*IX/n)

phi_rec = np.angle(Efinal)

# %% ============================================================
# FIDELITE
# ============================================================

Et = Etarget.reshape(-1); Et = Et / np.linalg.norm(Et)
Er = Efinal.reshape(-1);  Er = Er / np.linalg.norm(Er)
Fidelity = np.abs(np.vdot(Et, Er))**2

print('\n=====================================')
print(f'n         = {n}  ({n**2} miroirs/superpixel)')
print(f'r         = {r_norm:.4f}  < 1/(2n)={1/(2*n):.4f}')
print(f'Fidelite  = {Fidelity:.5f}')
print('=====================================')

# %% ============================================================
# RESULTATS
# ============================================================

fig = plt.figure('Reconstruction', figsize=(16, 8))

ax = fig.add_subplot(2, 4, 1)
im = ax.imshow(np.abs(Etarget), cmap='gray'); ax.axis('image'); fig.colorbar(im, ax=ax)
ax.set_title('Amplitude cible')

ax = fig.add_subplot(2, 4, 2)
im = ax.imshow(np.angle(Etarget), cmap='hsv'); ax.axis('image'); fig.colorbar(im, ax=ax)
ax.set_title('Phase cible')

ax = fig.add_subplot(2, 4, 3)
im = ax.imshow(np.abs(Efinal), cmap='gray'); ax.axis('image'); fig.colorbar(im, ax=ax)
ax.set_title('Amplitude reconstruite')

ax = fig.add_subplot(2, 4, 4)
im = ax.imshow(phi_rec, cmap='hsv'); ax.axis('image'); fig.colorbar(im, ax=ax)
ax.set_title('Phase reconstruite')

ax = fig.add_subplot(2, 4, 5)
im = ax.imshow(BinaryDevice, cmap='gray'); ax.axis('image'); fig.colorbar(im, ax=ax)
ax.set_title(f'Masque DMD ({Ndmd}x{Ndmd})')

ax = fig.add_subplot(2, 4, 6)
im = ax.imshow(np.angle(np.exp(1j*(np.angle(Etarget)-phi_rec))), cmap='hsv')
ax.axis('image'); fig.colorbar(im, ax=ax)
ax.set_title('Erreur de phase (enroulee)')

ax = fig.add_subplot(2, 4, 7)
ref = np.abs(Efinal)/np.max(np.abs(Efinal)) * np.max(np.abs(Etarget))
im = ax.imshow(np.abs(Etarget) - ref); ax.axis('image'); fig.colorbar(im, ax=ax)
ax.set_title("Erreur d'amplitude")

ax = fig.add_subplot(2, 4, 8)
ax.plot(Estate.real, Estate.imag, '.', markersize=2)
ax.plot(Etarget.reshape(-1).real, Etarget.reshape(-1).imag, 'r.', markersize=3)
ax.axis('equal'); ax.grid(True); ax.set_xlabel('Re(E)'); ax.set_ylabel('Im(E)')
ax.legend(['LUT', 'Cible'], loc='best')
ax.set_title('Plan complexe')

fig.suptitle(f'DMD superpixel [{target_mode}] — n={n} — theta={theta*180/np.pi:.0f} deg — '
             f'r={r_norm:.3f} — Fidelite={Fidelity:.4f}')
fig.tight_layout()

# %% ============================================================
# PSF — plan image (TF du champ pupille)
# ============================================================

pad_psf = 4
Npsf = pad_psf * Nsp

Et_pad = np.zeros((Npsf, Npsf), dtype=complex)
Er_pad = np.zeros((Npsf, Npsf), dtype=complex)
off_psf = (Npsf - Nsp) // 2
Et_pad[off_psf:off_psf+Nsp, off_psf:off_psf+Nsp] = Etarget
Er_pad[off_psf:off_psf+Nsp, off_psf:off_psf+Nsp] = Efinal

PSF_t = np.abs(np.fft.fftshift(np.fft.fft2(Et_pad)))**2; PSF_t /= np.max(PSF_t)
PSF_r = np.abs(np.fft.fftshift(np.fft.fft2(Er_pad)))**2; PSF_r /= np.max(PSF_r)

# Axes angulaires [lambda/D] ou D = Nsp * (d*n) = taille du champ de miroirs
dx_psf = 1 / (pad_psf * Nsp)                     # [cy/superpixel] par bin
u_ax = (np.arange(Npsf) - Npsf/2) * dx_psf * Nsp  # [lambda/D]

zoom_psf = 8

fig, axs = plt.subplots(1, 2, num='PSF reconstruite', figsize=(10, 4.8))
im0 = axs[0].imshow(10*np.log10(PSF_t + 1e-6),
                     extent=[u_ax[0], u_ax[-1], u_ax[0], u_ax[-1]],
                     origin='lower', cmap='hot', vmin=-40, vmax=0)
fig.colorbar(im0, ax=axs[0]); axs[0].axis('image')
axs[0].set_xlim(-zoom_psf, zoom_psf); axs[0].set_ylim(-zoom_psf, zoom_psf)
axs[0].set_xlabel(r'$\lambda/D$'); axs[0].set_ylabel(r'$\lambda/D$')
axs[0].set_title('PSF — cible')

im1 = axs[1].imshow(10*np.log10(PSF_r + 1e-6),
                     extent=[u_ax[0], u_ax[-1], u_ax[0], u_ax[-1]],
                     origin='lower', cmap='hot', vmin=-40, vmax=0)
fig.colorbar(im1, ax=axs[1]); axs[1].axis('image')
axs[1].set_xlim(-zoom_psf, zoom_psf); axs[1].set_ylim(-zoom_psf, zoom_psf)
axs[1].set_xlabel(r'$\lambda/D$'); axs[1].set_ylabel(r'$\lambda/D$')
axs[1].set_title('PSF — reconstruite (DMD)')

fig.suptitle(f'PSF [{target_mode}] — Fidelite={Fidelity:.4f}')
fig.tight_layout()

# --- Zoom masque ---
show = min(40, Nsp); npx = show*n
plt.figure('Masque DMD', figsize=(7, 7))
plt.imshow(BinaryDevice[:npx, :npx], cmap='gray')
plt.colorbar(); plt.axis('image')
plt.title(f'Masque DMD — {show}x{show} superpixels')
for k in range(show+1):
    plt.axvline(k*n - 0.5, color='b', linewidth=0.4, alpha=0.3)
    plt.axhline(k*n - 0.5, color='b', linewidth=0.4, alpha=0.3)

# %% ============================================================
# SAUVEGARDE MASQUE BINAIRE
# ============================================================

dmd_mask = BinaryDevice.astype(np.uint8)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
fid_str = f'{Fidelity:.4f}'.replace('.', 'p')
fname = f'DMD_mask_{target_mode}_n{n}_{fid_str}_r{r_norm:.3f}_{timestamp}.mat'
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
fpath = os.path.join(script_dir, fname)

savemat(fpath, {
    'dmd_mask': dmd_mask, 'n': n, 'Nsp': Nsp, 'Ndmd': Ndmd,
    'target_mode': target_mode, 'lam': lam, 'd': d, 'f_L2': f_L2,
    'f0x_norm': f0x_norm, 'f0y_norm': f0y_norm, 'r_norm': r_norm,
    'Fidelity': Fidelity, 'x_ap': x_ap, 'y_ap': y_ap, 'r_phys': r_phys,
})

print(f'\nMasque sauvegarde : {fpath}')
print(f'  Taille  : {Ndmd} x {Ndmd}  (uint8, 0/1)')
print(f'  Fidelite: {Fidelity:.4f}')

plt.show()