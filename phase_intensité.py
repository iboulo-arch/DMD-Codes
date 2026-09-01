import numpy as np
import matplotlib.pyplot as plt
import os

# %% Paramètres

# --- Dossier des mesures de phase (script courbe_amp_mode4) ---
base_dir_phase = r"C:\Users\ibrah\OneDrive\Bureau\stageLAM\Python_codes\Mesures_Stage_Lam\courbe m4_f_amp"
#base_dir_phase = r"C:\Users\ibrah\OneDrive\Bureau\stageLAM\Python_codes\Mesures_Stage_Lam\courbe m10_f_amp"
#base_dir_phase = r"C:\Users\ibrah\OneDrive\Bureau\stageLAM\Python_codes\Mesures_Stage_Lam\courbe m40_f_amp"
#base_dir_phase = r"C:\Users\ibrah\OneDrive\Bureau\stageLAM\Python_codes\Mesures_Stage_Lam\courbe m200_f_amp"

flat_file_phase = "flat_1.txt"
wavelength_nm = 632.8

# --- Dossier des mesures d'intensité (script intensity_plots) ---
base_dir_int = r"C:\Users\ibrah\OneDrive\Bureau\stageLAM\Python_codes\Mesures_Stage_Lam\Intensity_mode4"
#base_dir_int = r"C:\Users\ibrah\OneDrive\Bureau\stageLAM\Python_codes\Mesures_Stage_Lam\Intensity_mode10"
#base_dir_int = r"C:\Users\ibrah\OneDrive\Bureau\stageLAM\Python_codes\Mesures_Stage_Lam\Intensity_mode40"
#base_dir_int = r"C:\Users\ibrah\OneDrive\Bureau\stageLAM\Python_codes\Mesures_Stage_Lam\Intensity_mode200"

flat_file_int = "i-flat_1.txt"

# Liste (nom_fichier_phase, nom_fichier_intensite, amplitude d'entrée)
# L'ordre doit correspondre entre les deux jeux de fichiers
files_amp = [
    ("-01_2.txt", "i-01_2.txt", 0.1),
    ("-02_2.txt", "i-02_2.txt", 0.2),
    ("-03_2.txt", "i-03_2.txt", 0.3),
    ("-04_2.txt", "i-04_2.txt", 0.4),
    ("-05_2.txt", "i-05_2.txt", 0.5),
    ("-06_2.txt", "i-06_2.txt", 0.6),
    ("-07_2.txt", "i-07_2.txt", 0.7),
    ("-08_2.txt", "i-08_2.txt", 0.8),
    ("-09_2.txt", "i-09_2.txt", 0.9),
    ("-10_2.txt", "i-10_2.txt", 1.0),
    ("-14_2.txt", "i-14_2.txt", 1.4),
    ("-18_2.txt", "i-18_2.txt", 1.8),
    ("-24_2.txt", "i-24_2.txt", 2.4),
    ("-32_2.txt", "i-32_2.txt", 3.2),
    ("-40_2.txt", "i-40_2.txt", 4.0),
    ("-50_2.txt", "i-50_2.txt", 5.0),
    ("-100_2.txt", "i-100_2.txt", 10.0),
]


# %% Fonctions utilitaires

def load_rad(filepath, wavelength_nm, skiprows=12):
    """Charge un fichier de mesure de phase (en nm) et le convertit en radians RMS."""
    data_nm = np.loadtxt(filepath, skiprows=skiprows)
    return data_nm * (2 * np.pi / wavelength_nm)


def load_intensity(filepath, skiprows=12):
    """Charge un fichier de mesure d'intensité (pas de conversion d'unité)."""
    return np.loadtxt(filepath, skiprows=skiprows)


def rms(data):
    return np.sqrt(np.nanmean(data**2))


# %% Chargement des références "flat"

flat_phase = load_rad(rf"{base_dir_phase}\{flat_file_phase}", wavelength_nm)

flat_int_path = rf"{base_dir_int}\{flat_file_int}"
if not os.path.exists(flat_int_path):
    raise FileNotFoundError(f"Flat intensité introuvable : {flat_int_path}")
flat_int = load_intensity(flat_int_path)


# %% Boucle sur les fichiers : calcul RMS phase + RMS intensité, par rapport aux flats respectifs

x = []
y_phase = []      # amp_out en rad RMS (phase)
y_int = []        # RMS(diff intensité)
files_amp_used = []

for fname_phase, fname_int, amp_in in files_amp:
    filepath_phase = rf"{base_dir_phase}\{fname_phase}"
    filepath_int = rf"{base_dir_int}\{fname_int}"

    if not os.path.exists(filepath_phase):
        print(f"⚠️ Fichier phase manquant, ignoré : {filepath_phase}")
        continue
    if not os.path.exists(filepath_int):
        print(f"⚠️ Fichier intensité manquant, ignoré : {filepath_int}")
        continue

    data_phase = load_rad(filepath_phase, wavelength_nm)
    data_int = load_intensity(filepath_int)

    if data_int.shape != flat_int.shape:
        print(f"⚠️ Taille différente du flat intensité pour {fname_int}, ignoré : "
              f"{data_int.shape} vs {flat_int.shape}")
        continue

    rms_phase = rms(data_phase - flat_phase)
    rms_int = rms(data_int - flat_int)

    x.append(amp_in)
    y_phase.append(rms_phase)
    y_int.append(rms_int)
    files_amp_used.append((fname_phase, fname_int, amp_in))

x = np.array(x)
y_phase = np.array(y_phase)
y_int = np.array(y_int)


# %% Figure : amp_out (phase) + RMS intensité vs amp_in, sur deux axes y

fig, ax1 = plt.subplots()

color1 = "tab:blue"
ax1.set_xlabel(r"$amp_{in}$ (rad)")
ax1.set_ylabel(r"$amp_{out}$ (rad)", color=color1)
l1, = ax1.plot(x, y_phase, marker="o", color=color1, label=r"$amp_{out}$")
l_ref, = ax1.plot(x, x, linestyle="--", color="gray", label="1ère bissectrice (y=x)")
ax1.tick_params(axis="y", labelcolor=color1)
ax1.grid(True)

ax2 = ax1.twinx()
color2 = "tab:red"
ax2.set_ylabel(r"RMS intensité ($\Delta I$)", color=color2)
l2, = ax2.plot(x, y_int, marker="s", color=color2, label=r"RMS intensité ($\Delta I$)")
ax2.tick_params(axis="y", labelcolor=color2)

# Légende combinée des deux axes
lines = [l1, l_ref, l2]
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc="best")

plt.title("Comparaison des amplitudes de phase d'entrée et de sortie et RMS des intensités")
fig.tight_layout()
plt.show()