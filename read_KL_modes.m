%% ========================================================
% LOAD AND DISPLAY A KL MODE FROM HDF5
%% ========================================================

modal_basis_file = 'C:\Users\ibrah\OneDrive\Bureau\stageLAM\Python_codes\utc_2026-04-28_09-37-25_KL_modal_basis.h5';

% Load the modal basis  (dimensions: n_modes x Ny x Nx)
modal_basis = h5read(modal_basis_file, '/modal_basis');

% h5read returns data in Fortran order (x,y,mode) -> permute to (mode,y,x)
% to match Python's (mode, row, col) layout
modal_basis = permute(modal_basis, [3 2 1]);

%% Pupil mask : pixels that are non-zero in at least one mode
pupil = ~all(modal_basis == 0, 1);   % logical(Nsp x Nsp)
pupil = squeeze(pupil);

%% Pick the mode to display
mode_number = 450;   % same index as in Python (1-based in MATLAB -> +1)
KL_mode = squeeze(modal_basis(mode_number + 1, :, :));

%% Figures
figure
imagesc(pupil); axis image; colorbar
title('Pupil mask')

figure
imagesc(KL_mode); axis image; colorbar; colormap(hsv)
title(sprintf('KL Mode %d', mode_number))

%% Standard deviation per mode (over pupil pixels only)
n_modes = size(modal_basis, 1);
std_modes = zeros(n_modes, 1);
for k = 1:n_modes
    m = squeeze(modal_basis(k,:,:));
    std_modes(k) = std(m(pupil));
end

figure
plot(0:n_modes-1, std_modes)
xlabel('# mode'); ylabel('Standard deviation')
title('Standard Deviation of KL Modes')
grid on

modal_basis_raw = h5read(modal_basis_file, '/modal_basis');
disp(size(modal_basis_raw))