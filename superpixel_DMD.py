import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat

# scipy.special ou maoppy peuvent être utilisés pour Zernike
try:
    from maoppy.zernike import zernike, ansi2nm, ansi_name
    HAS_MAOPPY = True
except ImportError:
    HAS_MAOPPY = False

plt.close('all')

# %% ============================================================
# PARAMÈTRES
# ============================================================

Nsp = 100
n = 4
Ndmd = Nsp * n  # 400

lambda_w = 633e-9
d = 13.68e-6
theta = 12 * np.pi / 180

f_L2 = 300e-3

# Modes valides : 'zernike' | 'turbulence' | 'LG' | 'tiptilt' | 'focus' | 'flat'
target_mode = 'zernike'

# --- PARAMÈTRES DE ZERNIKE (Si target_mode = 'zernike') ---
j_zernike = 4  # Indice ANSI (4 = Defocus, 11 = Spherical, etc.)
amp_zernike = 1.0  # Amplitude en radians
use_zernike_combo = False
coeffs_zernike = {3: -1.0, 4: 1.5, 5: 0.5}

# --- Aberrations additionnelles ---
add_tiptilt = False
tip_rad = 2.0
tilt_rad = 3.1

add_focus = False
focus_rad = 1.0

amp_target = 1.0

# %% ============================================================
# GÉOMÉTRIE PHYSIQUE & OUVERTURE GOORDEN
# ============================================================

theta_illum = 2 * theta
theta_OFF_reflect = 2 * (-theta) - theta_illum  # -48 deg
x_OFF_phys = f_L2 * np.sin(theta_OFF_reflect)  # ~ -148 mm

f0x_norm = -1 / (n**2)  # -0.0625 cy/miroir
f0y_norm = +1 / n      # +0.25 cy/miroir

x_ap = f0x_norm * lambda_w * f_L2 / d
y_ap = f0y_norm * lambda_w * f_L2 / d

r_norm = 0.10
r_phys = r_norm * lambda_w * f_L2 / d

print(f"=== DMD ±{np.degrees(theta):.0f}° (Littrow {np.degrees(theta_illum):.0f}°) ===")
print(f"Pinhole sur le banc (plan focal L2, f={f_L2*1e3:.0f} mm) :")
print(f"  Centre : ({x_ap*1e3:.3f}, {y_ap*1e3:.3f}) mm")
print(f"  Rayon  : {r_phys*1e3:.3f} mm (diam. {2*r_phys*1e3:.3f} mm)")
print(f"  r norm : {r_norm:.4f} < 1/(2n)={1/(2*n):.4f} -> {'OK ✓' if r_norm < 1/(2*n) else 'FAIL ✗'}\n")

# %% ============================================================
# PORTEUR MIROIR OFF & NOYAU DFT
# ============================================================

f_off = 0.40

IC_g, IR_g = np.meshgrid(np.arange(Ndmd), np.arange(Ndmd))
phi_off_g = 2 * np.pi * (f_off * IC_g + f_off * IR_g)

ic_sp, ir_sp = np.meshgrid(np.arange(n), np.arange(n))
phi_off_sp = 2 * np.pi * (f_off * ic_sp + f_off * ir_sp)

demod_kernel = np.exp(-1j * 2 * np.pi * (f0x_norm * ic_sp + f0y_norm * ir_sp))

# %% ============================================================
# CIBLE & GÉNÉRATION DES MODES ZERNIKE (Échelle Superpixel)
# ============================================================

x_sp = np.linspace(-1, 1, Nsp)
Xn, Yn = np.meshgrid(x_sp, x_sp)
Rn = np.sqrt(Xn**2 + Yn**2)
pupil_mask = Rn <= 1.0

phi = np.zeros((Nsp, Nsp))
Etarget = None

mode = target_mode.lower()

if mode == 'zernike':
    print("Cible : Mode(s) de Zernike sur superpixels")
    if HAS_MAOPPY:
        if use_zernike_combo:
            for j, amp in coeffs_zernike.items():
                n_j, m_j = ansi2nm(j)
                Z_j = zernike(int(n_j), int(m_j), Nsp, norm="noll", outside=0)
                phi += amp * np.nan_to_num(Z_j)
        else:
            n_z, m_z = ansi2nm(j_zernike)
            Z = zernike(int(n_z), int(m_z), Nsp, norm="noll", outside=0)
            phi = amp_zernike * np.nan_to_num(Z)
    else:
        # Solution de repli synthétique si maoppy n'est pas installé (Defocus / Astigmatisme)
        print(" [Warning] maoppy non détecté, utilisation d'un polynôme analytique.")
        phi = amp_zernike * (2 * Rn**2 - 1)  # Z_4 Defocus par défaut
    
    phi = phi * pupil_mask

elif mode == 'turbulence':
    print("Cible : Kolmogorov turbulence")
    fx_t = np.fft.fftfreq(Nsp)
    FXt, FYt = np.meshgrid(fx_t, fx_t)
    ft = np.sqrt(FXt**2 + FYt**2)
    ft[ft == 0] = 1e-6
    PSD = ft**(-11/3)
    cn = np.random.randn(Nsp, Nsp) + 1j * np.random.randn(Nsp, Nsp)
    phi = np.real(np.fft.ifft2(np.fft.ifftshift(cn * np.sqrt(PSD))))
    phi = phi / np.std(phi) * np.pi

elif mode == 'lg':
    print("Cible : LG_1^0")
    PHI = np.arctan2(Yn, Xn)
    A = Rn * np.exp(-Rn**2)
    Etarget = (A / np.max(A)) * np.exp(1j * PHI)
    phi = np.angle(Etarget)

elif mode in ['tiptilt', 'focus', 'flat']:
    print(f"Cible : {mode}")
    phi = np.zeros((Nsp, Nsp))

# Superposition des aberrations additionnelles
phi_aberr = np.zeros((Nsp, Nsp))
if add_tiptilt:
    phi_aberr += tip_rad * Xn + tilt_rad * Yn
if add_focus:
    phi_aberr += focus_rad * (1 - Rn**2)

phi += phi_aberr

if Etarget is None:
    Etarget = np.exp(1j * phi)

# %% ============================================================
# CONSTRUCTION DE LA LOOK-UP TABLE (LUT)
# ============================================================

Nstates = 2**(n**2)
print(f"Génération LUT ({Nstates} états) ...")

Estate = np.zeros(Nstates, dtype=complex)
Bits = np.zeros((Nstates, n**2), dtype=bool)

for s in range(Nstates):
    bits = [(s >> b) & 1 for b in range(n**2)]
    bits_bool = np.array(bits, dtype=bool)
    Bits[s, :] = bits_bool
    block = bits_bool.reshape((n, n))
    Estate[s] = np.sum(block.astype(float) * demod_kernel)

Estate = Estate / np.max(np.abs(Estate))

A_opt = np.median(np.abs(Estate)) if amp_target == 'auto' else float(amp_target)
Etarget = A_opt * np.exp(1j * np.angle(Etarget))

# %% ============================================================
# DISPOSITION SUR DMD ET RECONSTRUCTION
# ============================================================

ComplexDevice = np.exp(1j * phi_off_g)
BinaryDevice = np.zeros((Ndmd, Ndmd), dtype=float)
Ereconstructed = np.zeros((Nsp, Nsp), dtype=complex)

for iy in range(Nsp):
    for ix in range(Nsp):
        target = Etarget[iy, ix]
        idx = np.argmin(np.abs(Estate - target))
        Ereconstructed[iy, ix] = Estate[idx]

        bits = Bits[idx, :]
        block = bits.reshape((n, n))

        ys, xs = iy * n, ix * n
        tile = np.exp(1j * phi_off_sp)
        tile[block] = 1.0

        ComplexDevice[ys:ys+n, xs:xs+n] = tile
        BinaryDevice[ys:ys+n, xs:xs+n] = block.astype(float)

# %% ============================================================
# PROPAGATION FOURIER COMPLÈTE (SIMULATION)
# ============================================================

Frecon = np.fft.fftshift(np.fft.fft2(ComplexDevice))
fx_r = np.fft.fftfreq(Ndmd)
FXr, FYr = np.meshgrid(fx_r, fx_r)

Hr = (((FXr - f0x_norm)**2 + (FYr - f0y_norm)**2) < r_norm**2).astype(float)
Ffilt = Frecon * Hr
ReconFull = np.fft.ifft2(np.fft.ifftshift(Ffilt))

# FFT Zero-padding pour affichage focal
pad_factor = 8
Npad = pad_factor * Ndmd
DevicePad = np.zeros((Npad, Npad), dtype=complex)
off_p = (Npad - Ndmd) // 2
DevicePad[off_p:off_p+Ndmd, off_p:off_p+Ndmd] = ComplexDevice

Ffull = np.fft.fftshift(np.fft.fft2(DevicePad))
fx_pad = np.fft.fftfreq(Npad)
FXp, FYp = np.meshgrid(fx_pad, fx_pad)

Hfull = (((FXp - f0x_norm)**2 + (FYp - f0y_norm)**2) < r_norm**2).astype(float)
Ffilt_disp = Ffull * Hfull

# %% ============================================================
# EXTRACTION DU CHAMP RECONSTRUIT ET FIDÉLITÉ
# ============================================================

IC_d, IR_d = np.meshgrid(np.arange(Ndmd), np.arange(Ndmd))
ReconDemod = ReconFull * np.exp(-1j * 2 * np.pi * (f0x_norm * IC_d + f0y_norm * IR_d))

Efinal = np.zeros((Nsp, Nsp), dtype=complex)
for iy in range(Nsp):
    for ix in range(Nsp):
        blk = ReconDemod[iy*n:(iy+1)*n, ix*n:(ix+1)*n]
        Efinal[iy, ix] = np.mean(blk)

IX, _ = np.meshgrid(np.arange(1, Nsp + 1), np.arange(1, Nsp + 1))
Efinal *= np.exp(-1j * 2 * np.pi * (IX - 1) / n)

# Calcul de la fidélité
Et = Etarget.ravel() / np.linalg.norm(Etarget.ravel())
Er = Efinal.ravel() / np.linalg.norm(Efinal.ravel())
Fidelity = np.abs(np.vdot(Et, Er))**2

print("=====================================")
print(f"Fidélité de reconstruction : {Fidelity:.5f}")
print("=====================================")

# %% ============================================================
# AFFICHAGE DES RÉSULTATS
# ============================================================

fig, axs = plt.subplots(2, 4, figsize=(16, 8))

axs[0, 0].imshow(np.abs(Etarget), cmap='gray'); axs[0, 0].set_title('Amplitude cible')
axs[0, 1].imshow(np.angle(Etarget), cmap='hsv'); axs[0, 1].set_title('Phase cible')
axs[0, 2].imshow(np.abs(Efinal), cmap='gray'); axs[0, 2].set_title('Amplitude reconstruite')
axs[0, 3].imshow(np.angle(Efinal), cmap='hsv'); axs[0, 3].set_title('Phase reconstruite')

axs[1, 0].imshow(BinaryDevice, cmap='gray'); axs[1, 0].set_title(f'Masque DMD ({Ndmd}x{Ndmd})')
axs[1, 1].imshow(np.angle(np.exp(1j * (np.angle(Etarget) - np.angle(Efinal)))), cmap='hsv')
axs[1, 1].set_title('Erreur phase (wrapped)')

ref = np.abs(Efinal) / np.max(np.abs(Efinal)) * np.max(np.abs(Etarget))
axs[1, 2].imshow(np.abs(Etarget) - ref, cmap='viridis'); axs[1, 2].set_title('Erreur amplitude')

axs[1, 3].plot(np.real(Estate), np.imag(Estate), '.', markersize=2)
axs[1, 3].plot(np.real(Etarget.ravel()), np.imag(Etarget.ravel()), 'r.', markersize=2)
axs[1, 3].set_aspect('equal'); axs[1, 3].grid(True); axs[1, 3].set_title('Plan complexe (LUT vs Target)')

plt.suptitle(f"Superpixel DMD [{target_mode}] - n={n} - Fidélité = {Fidelity:.4f}")
plt.tight_layout()
plt.show()

# %% ============================================================
# SAUVEGARDE MASQUE BINAIRE (.mat)
# ============================================================

mat_data = {
    'dmd_mask': BinaryDevice.astype(np.uint8),
    'n': n,
    'Nsp': Nsp,
    'Ndmd': Ndmd,
    'target_mode': target_mode,
    'Fidelity': Fidelity
}
savemat(f"DMD_mask_{target_mode}_n{n}.mat", mat_data)
print(f"Masque binaire sauvegardé sous : DMD_mask_{target_mode}_n{n}.mat")