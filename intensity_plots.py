import numpy as np
import matplotlib.pyplot as plt
import os

# %% Paramètres

base_dir = r"C:\Users\ibrah\OneDrive\Bureau\stageLAM\Python_codes\Mesures_Stage_Lam\Intensity_mode40"
#base_dir = r"C:\Users\ibrah\OneDrive\Bureau\stageLAM\Python_codes\Mesures_Stage_Lam\Intensity_mode200"

# Fichier de référence ("flat")
flat_file = "i-flat_1.txt"

files_amp = [
    ("i-01_2.txt", 0.1),
    ("i-02_2.txt", 0.2),
    ("i-03_2.txt", 0.3),
    ("i-04_2.txt", 0.4),
    ("i-05_2.txt", 0.5),
    ("i-06_2.txt", 0.6),
    ("i-07_2.txt", 0.7),
    ("i-08_2.txt", 0.8),
    ("i-09_2.txt", 0.9),
    ("i-10_2.txt", 1.0),
    ("i-14_2.txt", 1.4),
    ("i-18_2.txt", 1.8),
    ("i-24_2.txt", 2.4),
    ("i-32_2.txt", 3.2),
    ("i-40_2.txt", 4.0),
    ("i-50_2.txt", 5.0),
    ("i-100_2.txt", 10.0),
]

# %% Fonctions utilitaires

def load_intensity(filepath, skiprows=12):
    """Charge un fichier de mesure d'intensité (pas de conversion d'unité)."""
    data_int = np.loadtxt(filepath, skiprows=skiprows)
    return data_int


def rms(data):
    return np.sqrt(np.nanmean(data**2))


# %% Chargement de la référence "flat"

flat_path = rf"{base_dir}\{flat_file}"
if not os.path.exists(flat_path):
    raise FileNotFoundError(f"Flat introuvable : {flat_path}")

flat_int = load_intensity(flat_path)
rms_flat = rms(flat_int)

# %% Boucle sur les fichiers : calcul RMS + différence par rapport au flat

x = []
y = []
all_data_int = [flat_int]
files_amp_used = []

for fname, amp_in in files_amp:
    filepath = rf"{base_dir}\{fname}"
    if not os.path.exists(filepath):
        print(f"⚠️ Fichier manquant, ignoré : {filepath}")
        continue

    data_int = load_intensity(filepath)

    if data_int.shape != flat_int.shape:
        print(f"⚠️ Taille différente du flat pour {fname}, ignoré : {data_int.shape} vs {flat_int.shape}")
        continue

    rms_val = rms(data_int - flat_int)

    x.append(amp_in)
    y.append(rms_val)
    all_data_int.append(data_int)
    files_amp_used.append((fname, amp_in))

x = np.array(x)
y = np.array(y)

# %% Figure : amplitude en sortie vs amplitude en entrée

plt.figure()
plt.plot(x, y, marker="o", label="donnees")
plt.xlabel("amp_in")
plt.ylabel("RMS intensité (diff. par rapport au flat)")
plt.title("RMS intensité vs amplitude d'entrée")
plt.legend()
plt.grid(True)
plt.show()

# %% (Optionnel) Grille de cartes de différence par rapport au flat
"""
n = len(all_data_int) - 1
ncols = 3
nrows = int(np.ceil(n / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
axes = axes.flatten()

for i, data_int in enumerate(all_data_int[1:]):
    diff_int = data_int - flat_int
    im = axes[i].imshow(diff_int, cmap="jet")
    axes[i].set_title(f"a_in = {files_amp_used[i][1]} | rms = {y[i]:.3f}")
    plt.colorbar(im, ax=axes[i])

for j in range(n, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.show()"""