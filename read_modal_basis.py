# %%

import h5py
import matplotlib.pyplot as plt
import numpy as np

modal_basis_file_path = r"C:\Users\ibrah\OneDrive\Bureau\stageLAM\Python_codes\utc_2026-04-28_09-37-25_KL_modal_basis.h5"


with h5py.File(
    modal_basis_file_path,
    mode="r",
) as f:
    modal_basis = f["modal_basis"][...]

pupil = ~np.all(modal_basis == 0, axis=0)

mode_number = 10

plt.figure()
plt.imshow(pupil)
plt.title("Pupil")

plt.figure()
plt.imshow(modal_basis[mode_number])
plt.title(f"KL Mode {mode_number}")

# check that the standard deviation of the modes is unitary
plt.figure()
plt.plot(modal_basis[:, pupil].std(axis=1))
plt.title("Standard Deviation of KL Modes")
plt.xlabel("# mode")
plt.ylabel("Standard Deviation")

plt.show()
len(modal_basis)
# %%
