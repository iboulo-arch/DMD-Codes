# %%


import h5py
import matplotlib.pyplot as plt
import numpy as np

from aobench.utils.miscellaneous import get_utc_now
from aobench.display.live_view import live_view
from aobench.hardware.slm import MeadowlarkSLM
from aobench.hardware.camera import OrcaCamera, ThorlabsZeluxCamera
from aobench.hardware.connect import connect_hardware
from aobench.super_resolution import compute_cog
from aobench.config import Config

config = Config()

# %% Parameters

modal_basis_file_path = r"D:\Francois_Leroux\code\project\bioedge_bench\data\modal_basis\utc_2026-04-28_09-37-25_KL_modal_basis.h5"

dmd_phase_screen_file_path = r"D:\Francois_Leroux\code\project\bioedge_bench\data\internshib_ibrahima\KL_dataV2.h5"

exposure_time = 0.005  # seconds
n_frames_avg = 1000

# %% Read files

with h5py.File(
    modal_basis_file_path,
    mode="r",
) as f:

    modal_basis = f["modal_basis"][...]

with h5py.File(dmd_phase_screen_file_path, mode="r") as f:
    dmd_phase_screens = f["images"][...]

# %% inspect the modal bases

mode_number = 10
mode_number_dmd = 0

fig, axs = plt.subplots(1, 2, figsize=(15, 5))
axs[0].imshow(modal_basis[mode_number])
axs[0].set_title(f"KL Mode {mode_number}")
axs[1].imshow(dmd_phase_screens[mode_number_dmd])
axs[1].set_title(f"DMD Phase Screen {mode_number_dmd}")

# %% Check modal basis

pupil = ~np.all(modal_basis == 0, axis=0)

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

# %% Connect hardware

connect_slm_flag = True
connect_orca_flag = False
connect_thorcam_flag = True

slm: MeadowlarkSLM | None
orca: OrcaCamera | None
thorcam: ThorlabsZeluxCamera | None

slm, orca, thorcam = connect_hardware(
    config,
    connect_slm_flag=connect_slm_flag,
    connect_orca_flag=connect_orca_flag,
    connect_thorcam_flag=connect_thorcam_flag,
)

# %% Set exposure time

thorcam.exposure_time = exposure_time

# %% Display live view of thorcam

live_view(thorcam)

# %% set thorcam ROI

thorcam.reset_roi()  # reset roi to full sensor
mean_frame = thorcam.acquire_mean(100)
cog_y, cog_x = compute_cog(mean_frame)
cog_y, cog_x = np.unravel_index(np.argmax(mean_frame), mean_frame.shape)

npx = 100

# %%

thorcam.roi = (int(cog_x - npx), int(cog_y - npx), int(cog_x + npx), int(cog_y + npx))

# %% Acquire dark frame - Turn off ligth source

dark = thorcam.acquire_mean(n_frames_avg)

# %% Acquire reference psf

slm.reset_slm_phase()

# %%

reference_psf = thorcam.acquire_mean(n_frames_avg) - dark

# %% Display reference phase on SLM

stroke = 1  # [rad]

slm.display_phase(255 / (2 * np.pi) * stroke * modal_basis[mode_number], display=True)

# %% Display live view of thorcam

live_view(thorcam)

# %% Acquire mean frame

mean_frame_classical_slm = thorcam.acquire_mean(n_frames_avg) - dark

# %% Display DMD phase on SLM

slm.display_phase(dmd_phase_screens[mode_number_dmd], display=True)

# %% Display live view of thorcam

live_view(thorcam)

# %% Acquire mean frame

mean_frame_dmd = thorcam.acquire_mean(n_frames_avg) - dark

# %% Save results

utc_now = get_utc_now()

with h5py.File(
    r"D:\Francois_Leroux\code\project\bioedge_bench\data\internshib_ibrahima\\"
    f"{utc_now}_test_focal_plane_camera.h5",
    mode="w",
) as f:
    f.attrs["exposure_time"] = exposure_time
    f.attrs["n_frames_avg"] = n_frames_avg
    f.create_dataset(
        "command_slm_classic",
        data=255 / (2 * np.pi) * stroke * modal_basis[mode_number],
    )
    f.create_dataset("command_slm_dmd", data=dmd_phase_screens[mode_number_dmd])
    f.create_dataset("dark", data=dark)
    f.create_dataset("mean_reference_psf", data=reference_psf)
    f.create_dataset("mean_frame_classical_slm", data=mean_frame_classical_slm)
    f.create_dataset("mean_frame_dmd", data=mean_frame_dmd)
