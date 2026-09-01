# %%
import numpy as np
import matplotlib.pyplot as plt

import tifffile as tiff
import numpy as np



# %% Chargement des données HASO

#file1 = r"C:\Users\ibrah\OneDrive\Bureau\stageLAM\Python_codes\Mesures_Stage_Lam\haso_lo\HASO_2_4_nh.txt"
#file1 = r"C:\Users\ibrah\OneDrive\Bureau\stageLAM\Python_codes\Mesures_Stage_Lam\haso_lo\HASO_2_40_nh.txt"
#file1 = r"C:\Users\ibrah\OneDrive\Bureau\stageLAM\Python_codes\Mesures_Stage_Lam\haso_lo\HASO_0_nh.txt"
file1 = r"C:\Users\ibrah\OneDrive\Bureau\stageLAM\Python_codes\Mesures_Stage_Lam\haso_lo\HASO_2_200_nh.txt"
file2 = r"C:\Users\ibrah\OneDrive\Bureau\stageLAM\Python_codes\Mesures_Stage_Lam\haso_lo\HASO_1_nh.txt"

data1 = np.loadtxt(file1, skiprows=12)  # car  en-tête
data2 = np.loadtxt(file2, skiprows=12)
diff = data1 - data2

wavelength_nm = 632.8 

data1_rad = data1 * (2 * np.pi / wavelength_nm)
data2_rad = data2 * (2 * np.pi / wavelength_nm)
diff_rad  = data1_rad - data2_rad   # ou diff * (2*np.pi/wavelength_nm), équivalent

# RMS en rad pour voir les amplitudes
rms_data1 = np.sqrt(np.nanmean(data1_rad**2))
rms_data2 = np.sqrt(np.nanmean(data2_rad**2))
rms_diff  = np.sqrt(np.nanmean(diff_rad**2))

print(rms_diff)


print("Shape data1:", data1.shape)
print("Shape data2:", data2.shape)
print("Shape diff:", diff.shape)

# %% Chargement des données PSF

file3 = r"C:\Users\ibrah\OneDrive\Bureau\stageLAM\Python_codes\Mesures_Stage_Lam\psf_lo\PSF_1_nd.tif"
file4 = r"C:\Users\ibrah\OneDrive\Bureau\stageLAM\Python_codes\Mesures_Stage_Lam\psf_lo\PSF_1_d.tif"


img3 = tiff.imread(file3)
img4 = tiff.imread(file4)

# %% Affichage côte à côte
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

im1 = axes[1].imshow(data1_rad, cmap='jet') # question sur ces resultats, on a jamais mis le rms alors qu'on était censé les mettre en radRMS
axes[1].set_title(r"$Mode_{inj}$ + $Flat_1$")
plt.colorbar(im1, ax=axes[1], label='Phase (rad)')

im2 = axes[0].imshow(data2_rad, cmap='jet',)
axes[0].set_title(r"$Flat_1$")
plt.colorbar(im2, ax=axes[0], label='Phase (rad)')

im3 = axes[2].imshow(diff_rad, cmap='jet')
axes[2].set_title(r"Reel $mode_{inj}$")
plt.colorbar(im3, ax=axes[2], label='Phase (rad)')
plt.tight_layout()


"""fig, axes = plt.subplots(1, 3, figsize=(18, 5))
im1 = axes[0].imshow(img3, cmap='gray')
axes[0].set_title("PSF 1_nd")
im2 = axes[1].imshow(img4, cmap='gray')
axes[1].set_title("PSF 1_d")

im3 = axes[2].imshow(img3 - img4, cmap='gray')
axes[2].set_title("PSF 1_nd - PSF 1_d")

"""
plt.show()