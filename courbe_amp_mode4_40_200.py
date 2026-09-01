import numpy as np
import matplotlib.pyplot as plt
import os

# %% Paramètres

base_dir = r"C:\Users\ibrah\OneDrive\Bureau\stageLAM\Python_codes\Mesures_Stage_Lam\courbe m4_f_amp"
#base_dir = r"C:\Users\ibrah\OneDrive\Bureau\stageLAM\Python_codes\Mesures_Stage_Lam\courbe m40_f_amp"
#base_dir = r"C:\Users\ibrah\OneDrive\Bureau\stageLAM\Python_codes\Mesures_Stage_Lam\courbe m200_f_amp"


wavelength_nm = 632.8

# Fichier de référence ("flat")
flat_file = "flat_1.txt"

# Liste (nom_fichier, amplitude d'entrée correspondante)
# L'ordre doit correspondre à x = [0.1, 0.2, ..., 10]
files_amp = [
    ("-01_2.txt", 0.1),
    ("-02_2.txt", 0.2),
    ("-03_2.txt", 0.3),
    ("-04_2.txt", 0.4),
    ("-05_2.txt", 0.5),
    ("-06_2.txt", 0.6),
    ("-07_2.txt", 0.7),
    ("-08_2.txt", 0.8),
    ("-09_2.txt", 0.9),
    ("-10_2.txt", 1.0),
    ("-14_2.txt", 1.4),
    ("-18_2.txt", 1.8),
    ("-24_2.txt", 2.4),
    ("-32_2.txt", 3.2),
    ("-40_2.txt", 4.0),
    ("-50_2.txt", 5.0),
    ("-100_2.txt", 10.0),
]

# %% Fonctions utilitaires

def load_rad(filepath, wavelength_nm, skiprows=12):
    """Charge un fichier de mesure (en nm) et le convertit en radians RMS."""
    data_nm = np.loadtxt(filepath, skiprows=skiprows)
    data_rad = data_nm * (2 * np.pi / wavelength_nm)
    return data_rad


def rms(data_rad):
    return np.sqrt(np.nanmean(data_rad**2))


# %% Chargement de la référence "flat"

flat_rad = load_rad(rf"{base_dir}\{flat_file}", wavelength_nm)
#rms_flat = rms(flat_rad)

# %% Boucle sur les fichiers : calcul RMS + différence par rapport au flat

x = []
y = []
all_data_rad = [flat_rad]
files_amp_used = []  # pour garder la trace de ce qui a effectivement été chargé

for fname, amp_in in files_amp:
    filepath = rf"{base_dir}\{fname}"
    if not os.path.exists(filepath):
        print(f"⚠️ Fichier manquant, ignoré : {filepath}")
        continue

    data_rad = load_rad(filepath, wavelength_nm)
    rms_val = rms(data_rad - flat_rad)

    x.append(amp_in)
    y.append(rms_val)
    all_data_rad.append(data_rad)
    files_amp_used.append((fname, amp_in))

x = np.array(x)
y = np.array(y)

# %% Figure : amplitude en sortie vs amplitude en entrée

plt.figure()
plt.plot(x, y, marker="o", label="Mesures")
plt.plot(x, x, linestyle="--", color="gray", label="1ere bissectrice (y=x)")
plt.xlabel(r"$amp_{in}$ (rad)")
plt.ylabel(r"$amp_{out}$ (rad)")
plt.title("Comparaison des amplitudes d'entrée et de sortie")
plt.legend()
plt.grid(True)
plt.show()

# %% (Optionnel) Grille de cartes de différence par rapport au flat
"""""
n = len(all_data_rad) - 1  # nombre de mesures comparées au flat
ncols = 3
nrows = int(np.ceil(n / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
axes = axes.flatten()

for i, data_rad in enumerate(all_data_rad[1:]):
    diff_rad = data_rad - flat_rad
    im = axes[i].imshow(diff_rad, cmap="jet")
    axes[i].set_title(f"a_in = {files_amp_used[i][1]} | a_out = {y[i]:.3f}")
    plt.colorbar(im, ax=axes[i])

for j in range(n, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.show()
"""