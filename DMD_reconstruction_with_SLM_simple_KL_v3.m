%% ========================================================
% SLM SUPERPIXEL — DIRECT PHASE ENCODING
%
% Simplified approach: the SLM directly encodes the superpixel
% phase map.  Each SLM pixel = one virtual DMD micro-mirror.
%
%   Superpixel (4x4 pixels):
%   ┌───────────────────────────┐
%   │  0   π/8  2π/8  3π/8     │
%   │ 4π/8 5π/8 6π/8  7π/8     │
%   │ 8π/8 9π/8 10π/8 11π/8    │
%   │12π/8 13π/8 14π/8 15π/8   │
%   └───────────────────────────┘
%
%   ON  mirror at (iy2,ix2)  ->  SLM pixel phase = phases(iy2,ix2)
%   OFF mirror               ->  SLM pixel phase = phi_off
%
% Since the phase of each pixel is directly programmed, the aperture
% is placed at DC (centred, NO decentering needed).
% The aperture acts only as a low-pass spatial filter.
%
% Comparison with the blaze-based approach (DMD_reconstruction_with_SLM):
%   - Simpler : no SLM_per_DMD, no carrier correction, no decentering
%   - Less efficient : OFF pixels still reach the aperture (phase_off ≠ 0)
%   - Direct : the LUT IS the set of complex phasors summed by the aperture
%% ========================================================

clear; close all; clc;

%% ========================================================
% PARAMETERS
%% ========================================================

Nsp    = 100;        % number of superpixels per axis
n      = 4;          % superpixel size (n x n SLM pixels = n x n mirrors)
lambda = 633e-9;     % wavelength [m]

SLM_pitch = 9.2e-6;  % SLM pixel pitch [m] (info only, not used in sim)

%% ========================================================
% APERTURE  (centred at DC, no decentering)
%% ========================================================

r = 0.090;    % radius [normalised frequency units, cycles/pixel]
              % rule of thumb : r < 1/(2*n) = 0.125 so individual
              % pixels cannot be resolved (superpixel acts as a unit)
f_L2=200e-3;
r_physique = r * lambda * f_L2 / (SLM_pitch*1000); %mm
         %  = 0.48 mm  %(rayon du pinhole à insérer dans ton banc)              
%% ========================================================
% OFF-PIXEL PHASE
%
% For a pure phase SLM, OFF pixels cannot be truly blocked.
% phi_off = 0       : OFF pixels add a real offset to the field
%                     (shifts the LUT cloud toward +Re axis)
% phi_off = random  : OFF contributions average toward zero
%                     (cleaner LUT, set use_random_off = true)
%% ========================================================
% gestion des pixels OFF.
use_random_off = true;   % set true for random OFF phases

if use_random_off
    rng(50)   % graine fixe. si le chiffre est changé, les valeurs randoms d'un superpixel changent.
    phi_off_map = 2*pi * rand(n,n);   % random per mirror, fixed for LUT
else
    phi_off_map = zeros(n,n);         % all OFF -> phase 0
end

%% ========================================================
% SUPERPIXEL PHASE MAP  (the 16 uniformly distributed phases)
%
%   phases(iy2,ix2) = 2*pi * k / n^2   with k = 0..n^2-1
%   arranged in column-major order (MATLAB default for reshape).
%
%   Neighbours in y  : delta_phi = 2*pi / n^2  (= pi/8  for n=4)
%   Neighbours in x  : delta_phi = 2*pi / n    (= pi/2  for n=4)
%% ========================================================

phases = reshape(2*pi*(0:n^2-1)/n^2, n, n); %répartition uniforme des phases sur un superpixel

fprintf('Phase map (multiples of pi/8):\n')
disp(round(phases / (pi/8)))

%% ========================================================
% TARGET FIELD
%
% Choose one of the targets below.
% The LUT spans a DISK in the complex plane, so both amplitude
% and phase can be controlled simultaneously.
%% ========================================================

% target_mode = 'turbulence';
% target_mode = 'turbulence';   % pure phase only, |E|=1
% target_mode = 'image';        % arbitrary amplitude + phase
 target_mode = 'kl';

%% --- Kolmogorov turbulence (pure phase, |E| = 1) ---
fx_t = (-Nsp/2:Nsp/2-1) / Nsp;
[FXt, FYt] = meshgrid(fx_t, fx_t);
ft = sqrt(FXt.^2 + FYt.^2);
ft(ft==0) = 1e-6;
PSD = ft.^(-11/3);
cn  = randn(Nsp) + 1i*randn(Nsp);
phi_turb = real(ifft2(ifftshift(cn .* sqrt(PSD))));
phi_turb = phi_turb / std(phi_turb(:)) * pi;

%% --- Laguerre-Gaussian LG_l^p (amplitude + phase) ---
l = 1; p = 0;   % azimuthal index, radial index
[X, Y] = meshgrid(linspace(-1,1,Nsp), linspace(-1,1,Nsp));
R   = sqrt(X.^2 + Y.^2);
PHI = atan2(Y, X);
% Amplitude: R^|l| * exp(-R^2) * Laguerre_p^|l|(2R^2)  (p=0: Laguerre=1)
A_LG = R.^abs(l) .* exp(-R.^2);
A_LG = A_LG / max(A_LG(:));
E_LG = A_LG .* exp(1i*l*PHI);

%% --- KL mode (à charger AVANT le switch) ---
mode_number = 10;   % indice Python 0-based

%% ===== CONTROLE STD DES MODES KL =====
% gain_std = 1  -> std naturel du mode KL
% gain_std > 1 -> augmente la turbulence de ce mode
% gain_std < 1 -> diminue la turbulence de ce mode

gain_std = 1;

% exemple :
% gain_std = 2.0;   % double le std
% gain_std = 0.5;   % divise le std par 2

%% =====================================
modal_basis_file = 'C:\Users\ibrah\OneDrive\Bureau\stageLAM\Python_codes\utc_2026-04-28_09-37-25_KL_modal_basis.h5';
modal_basis = h5read(modal_basis_file, '/modal_basis');
modal_basis = permute(modal_basis, [3 2 1]);   % adapter si besoin

%% --- Select target ---
switch lower(target_mode)
    case 'turbulence'
        fprintf('Target: Kolmogorov turbulence (phase only)\n')
        phi      = phi_turb;
        Etarget  = exp(1i*phi);

    case 'lg'
        fprintf('Target: LG_%d^%d mode (amplitude + phase)\n', l, p)
        phi      = angle(E_LG);
        Etarget  = E_LG;              % complex amplitude, |E| varies!
        Etarget  = Etarget / max(abs(Etarget(:)));

    case 'image'
        fprintf('Target: image (amplitude + phase)\n')
        % Load any grayscale image as amplitude, turbulence as phase
        % (replace 'your_image.png' with your file)
         %img = double(rgb2gray(imread('images.png')));
         cdata=imread('Sans titre.jpg');
         img=double(rgb2gray((cdata)));
         img = imresize(img, [Nsp Nsp]);
         img = img / max(img(:));
         Etarget = double(img).* exp(1i*phi_turb);
       % error('Set your image path in the ''image'' case.')

    case 'kl'
        fprintf('Target: KL mode %d\n', mode_number)
    
        % Charger le mode KL original
        KL_mode = squeeze(modal_basis(mode_number + 1, :, :));
    
        % ===== CALCUL DU STD ICI =====
        pupil = KL_mode ~= 0;
    
        std_original = std(KL_mode(pupil));
    
        fprintf('KL mode %d : std original = %.6f\n', ...
            mode_number, std_original);
        % =============================
    
    
        % Redimensionnement
        KL_mode = imresize(KL_mode, [Nsp Nsp]);
    
    
        % ======================================================
        % APPLICATION DU STD VOULU
        % ======================================================

        pupil = KL_mode ~= 0;
        
        std_before = std(KL_mode(pupil));
        
        % changement de variance du mode
        KL_mode = KL_mode * (gain_std);
        
        std_after_gain = std(KL_mode(pupil));
        
        fprintf('KL %d : std avant = %.6f\n', ...
            mode_number, std_before);
        
        fprintf('KL %d : std après gain = %.6f\n', ...
            mode_number, std_after_gain);
        
        
        % Normalisation uniquement pour garder une phase exploitable
        KL_mode = KL_mode / max(abs(KL_mode(:)));
        
        % vérifier
        std_final = std(KL_mode(KL_mode~=0));
        
        fprintf('KL %d : std final après normalisation = %.6f\n',...
            mode_number,std_final);

        % (optionnel) voir le std après ta normalisation
        pupil2 = KL_mode ~= 0;
        std_after = std(KL_mode(pupil2));
    
        fprintf('KL mode %d : std après max-normalisation = %.6f\n', ...
            mode_number, std_after);
    
        Etarget = exp(1i*KL_mode);
       % Etarget = abs(KL_mode) .* exp(1i * pi * (KL_mode < 0));


    otherwise
        error('Unknown target_mode: "%s". Use ''turbulence'', ''LG'' or ''image''.', target_mode)
end

%% ========================================================
% BUILD PHYSICAL LUT
%
% For each of the 2^(n^2) binary patterns:
%   complex_block(iy2,ix2) = exp(i*phases(iy2,ix2))   if ON
%                           = exp(i*phi_off(iy2,ix2))  if OFF
%
% The aperture at DC sums contributions from all pixels.
% No zero-padding is strictly needed here (the block is already
% small), but we keep it for numerical accuracy.
%% ========================================================

fprintf('Building physical LUT...\n')

K = 50000;
Estate  = zeros(K, 1);
Bits    = false(K, n^2);

pad = 128;

fx_lut = (-pad/2:pad/2-1) / pad;
[FX_lut, FY_lut] = meshgrid(fx_lut, fx_lut);

%% Centred aperture
H = double(sqrt(FX_lut.^2 + FY_lut.^2) < r);

%% OFF pixel complex field (fixed for all states)
off_block = exp(1i*phi_off_map);   % n x n

for s = 1:K
    bits        = rand(1, n^2) > 0.5;
    Bits(s,:)   = bits;
    block       = reshape(bits, n, n);

    complex_block = off_block;
    complex_block(block) = exp(1i*phases(block));

    tmp = zeros(pad);
    c0x = floor(pad/2 - n/2) + 1;
    c0y = floor(pad/2 - n/2) + 1;
    tmp(c0y:c0y+n-1, c0x:c0x+n-1) = complex_block;

    F     = fftshift(fft2(tmp));
    Ffilt = F .* H;
    recon = ifft2(ifftshift(Ffilt));

    Estate(s) = recon(pad/2+1, pad/2+1);
end

%% Normalize LUT
Estate = Estate / max(abs(Estate));

%% ========================================================
% DISPLAY LUT
%% ========================================================

figure
plot(real(Estate), imag(Estate), '.')
axis equal; grid on
xlabel('Re(E)'); ylabel('Im(E)')
title(sprintf('Direct SLM LUT  (n=%d, r=%.3f, phi_off=%s)', ...
    n, r, ternary_str(use_random_off,'random','0')))

%% ========================================================
% BUILD DEVICE  (Nsp*n x Nsp*n SLM pixels)
%% ========================================================

fprintf('Building device...\n')

Ndmd   = Nsp * n;
Device = zeros(Ndmd, Ndmd);    % stores SLM phase values

%% ========================================================
% RECONSTRUCTION
%% ========================================================

Ereconstructed = zeros(Nsp, Nsp);

for iy = 1:Nsp
    for ix = 1:Nsp

        target  = Etarget(iy,ix);
        [~,idx] = min(abs(Estate - target));
        Ereconstructed(iy,ix) = Estate(idx);

        bits  = Bits(idx,:);
        block = reshape(bits, n, n);

        ys = (iy-1)*n + 1;
        xs = (ix-1)*n + 1;

        %% Fill the SLM phase map for this superpixel
        tile = phi_off_map;               % OFF pixels
        tile(block) = phases(block);      % ON  pixels get their phase

        Device(ys:ys+n-1, xs:xs+n-1) = tile;

    end
end

%% ========================================================
%% FULL FOURIER PROPAGATION (avec zero-padding)- and PSF
%% ========================================================
ComplexDevice = exp(1i*Device);
% Masque pupille circulaire

[XX, YY] = meshgrid((-Ndmd/2:Ndmd/2-1), (-Ndmd/2:Ndmd/2-1));
pupil_mask = double(sqrt(XX.^2 + YY.^2) <= Ndmd/2);
ComplexDevice = ComplexDevice.* pupil_mask;
pad_factor = 20;
Npad = Ndmd * pad_factor;

ComplexDevice_pad = zeros(Npad, Npad);
offset = round((Npad - Ndmd) / 2);
ComplexDevice_pad(offset+1:offset+Ndmd, offset+1:offset+Ndmd) = ComplexDevice;

% Filtre sur la grille paddée
fx_pad = (-Npad/2:Npad/2-1) / Npad;
[FX_pad, FY_pad] = meshgrid(fx_pad, fx_pad);
Hpad = double(sqrt(FX_pad.^2 + FY_pad.^2) < r);

% FFT paddée
F_pad   = fftshift(fft2(ComplexDevice_pad));
Ffilt   = F_pad .* Hpad;
ReconFull_pad = ifft2(ifftshift(Ffilt));

% PSF
PSF = abs(Ffilt).^2 ;
PSF = PSF / max(PSF(:));

% Axe angulaire corrigé (Npad, pas Ndmd)
grandissement = 100/200;  % f_L3/f_L2;
angle_per_pixel_matlab = lambda / (SLM_pitch / grandissement) / Npad;  % rad/pixel
ax_matlab_urad = (-Npad/2:Npad/2-1) * angle_per_pixel_matlab * 1e6;

%% ========================================================
% EXTRACT RECONSTRUCTED FIELD
%% ========================================================

Efinal = zeros(Nsp, Nsp);
for iy = 1:Nsp
    for ix = 1:Nsp
        ys = offset + (iy-1)*n + 1;
        xs = offset + (ix-1)*n + 1;
        blk = ReconFull_pad(ys:ys+n-1, xs:xs+n-1);
        Efinal(iy,ix) = mean(blk(:));
    end
end

%% ========================================================
% FIDELITY
%% ========================================================

phi_rec = angle(Efinal);

Et = Etarget(:);  Et = Et / norm(Et);
Er = Efinal(:);   Er = Er / norm(Er);

Fidelity = abs(Et'*Er)^2;

fprintf('\n')
fprintf('=====================================\n')
fprintf('n         = %d  (%d pixels/superpixel)\n', n, n^2)
fprintf('r         = %.4f\n', r)
fprintf('phi_off   = %s\n', ternary_str(use_random_off,'random','0'))
fprintf('Fidelity  = %.5f\n', Fidelity)
fprintf('=====================================\n')

%% ========================================================
% DISPLAY
%% ========================================================

figure('Position',[100 100 1800 900])

subplot(2,4,1) 
imagesc(abs(Etarget)); axis image; colorbar; colormap(gray)
title('Target amplitude')

subplot(2,4,2)
imagesc(angle(Etarget)); axis image; colorbar; colormap(hsv)
title('Target phase')

subplot(2,4,3)
imagesc(abs(Efinal)); axis image; colorbar; colormap(gray)
title('Reconstructed amplitude')

subplot(2,4,4)
imagesc(phi_rec); axis image; colorbar; colormap(hsv)
title('Reconstructed phase')

subplot(2,4,5)
imagesc(abs(Etarget) - abs(Efinal)/max(abs(Efinal(:)))*max(abs(Etarget(:))))
axis image; colorbar; colormap(gray)
title('Amplitude error')

subplot(2,4,6)
imagesc(angle(exp(1i*(angle(Etarget) - phi_rec)))); axis image; colorbar; colormap(hsv)
title('Wrapped phase error')

subplot(2,4,7)
imagesc(Device); colormap(hsv)
axis image; colorbar
title('SLM device (direct phase)')

subplot(2,4,8)
plot(real(Estate), imag(Estate), '.', 'MarkerSize', 2)
hold on
plot(real(Etarget(:)), imag(Etarget(:)), 'r.', 'MarkerSize', 3)
axis equal; grid on
xlabel('Re(E)'); ylabel('Im(E)')
legend('LUT achievable', 'Target points')
title('Complex plane')

sgtitle(sprintf('Direct SLM [%s] — n=%d — r=%.3f — Fidelity = %.4f', ...
    target_mode, n, r, Fidelity))

% ========================================================
% SLM MAP  (8-bit output, 256 gray levels = 256 phase values)
% 
% The SLM accepts a grayscale image where:
%   gray level 0   -> phase 0
%   gray level 255 -> phase 2*pi  (or 2*pi*(255/256) depending on SLM)
% 
% Device contains phase values in [0, 2*pi).
% Conversion: gray = round( mod(phase, 2*pi) / (2*pi) * 255 )
%% ========================================================

SLM_map = uint8(round(mod(Device, 2*pi) / (2*pi) * 255));

figure('Position',[100 100 700 700])
imagesc(SLM_map, [0 255])
colormap(gray)
colorbar
axis image
title(sprintf('SLM map — %d x %d pixels — 8 bit (0..255)', ...
    size(SLM_map,2), size(SLM_map,1)))
xlabel('x [SLM pixels]')
ylabel('y [SLM pixels]')

% Zoom on one superpixel to check the 4x4 phase pattern
figure('Position',[850 100 400 400])
imagesc(SLM_map(1:n, 1:n), [0 255])
colormap(gray)
colorbar
axis image
title('Zoom: first superpixel (4x4 pixels)')
for iy2 = 1:n
    for ix2 = 1:n
        text(ix2, iy2, num2str(SLM_map(iy2,ix2)), ...
            'HorizontalAlignment','center', ...
            'Color','r', 'FontWeight','bold', 'FontSize',10)
    end
end

fprintf('\nSLM map: %d x %d pixels, uint8 [0..255]\n', ...
    size(SLM_map,2), size(SLM_map,1))
fprintf('Phase resolution: 2*pi / 256 = %.4f rad/level\n', 2*pi/256)

% Optional: save to file
% imwrite(SLM_map, 'SLM_map_simple.png')

%% ========================================================
% LOCAL HELPER
%% ========================================================

function s = ternary_str(cond, a, b)
    if cond, s = a; else, s = b; end
end


%% ========================================================
% EXPORT POUR LE SLM MEADOWLARK (1920x1152)
%% ========================================================

% Redimensionner à la résolution physique du SLM
SLM_map_full = imresize(SLM_map, [660 660], 'nearest');

%output_path = sprintf('C:\\Users\\ibrah\\OneDrive\\Bureau\\stageLAM\\SLM_phase_map2\\KL%d_Nsp%d_n%d_F%.3f_std.%2f.bmp', ...
  %  mode_number, Nsp, n, Fidelity, gain_std);

%imwrite(SLM_map_full, output_path);
%fprintf('Sauvegardé : %s\n', output_path)

%fprintf('Image sauvegardée : %s\n', output_path)
%fprintf('Dimensions : %d x %d pixels, 8-bit\n', size(SLM_map_full,2), size(SLM_map_full,1))

%% ========================================================
% PSF — affichage
%% ========================================================

% ax_matlab_urad déjà correct (basé sur Npad)
fx_psf = (-Npad/2:Npad/2-1) / Npad;

figure('Position',[100 100 1200 500])

subplot(1,2,1)
imagesc(ax_matlab_urad, ax_matlab_urad, log10(PSF + 1e-6))
axis image; colorbar; colormap(hot)
title('PSF (log_{10}) — µrad')
xlabel('angle [µrad]'); ylabel('angle [µrad]')
xlim([-1500 1500]); ylim([-1500 1500])

subplot(1,2,2)
imagesc(fx_psf, fx_psf, log10(PSF + 1e-6))
axis image; colorbar; colormap(hot)
title('PSF (log_{10}) — cycles/pixel')
xlabel('f_x'); ylabel('f_y')
xlim([-5*r 5*r]); ylim([-5*r 5*r])

sgtitle(sprintf('PSF — n=%d — Nsp=%d — r=%.4f', n, Nsp, r))

%% Export PSF simulée pour Python
pitch_camera      = 3.45e-6;
f_L4bis           = 200e-3;
sampling_sim_urad = angle_per_pixel_matlab * 1e6;        % µrad/pixel (Npad)
sampling_cam_urad = pitch_camera / f_L4bis * 1e6;        % 17.25 µrad/pixel

fprintf('sampling_sim  = %.4f µrad/pixel\n', sampling_sim_urad)
fprintf('sampling_cam  = %.4f µrad/pixel\n', sampling_cam_urad)

factor = sampling_sim_urad / sampling_cam_urad;
fprintf('factor        = %.4f\n', factor)

Ny_psf   = size(PSF, 1);
Nx_psf   = size(PSF, 2);
Ny_target = max(1, round(Ny_psf * factor));
Nx_target = max(1, round(Nx_psf * factor));

fprintf('PSF size      = %d x %d\n', Ny_psf, Nx_psf)
fprintf('Target size   = %d x %d\n', Ny_target, Nx_target)

PSF_resampled = imresize(PSF, [Ny_target Nx_target], 'bicubic');
PSF_resampled = PSF_resampled / max(PSF_resampled(:));

ax_resampled_urad = (-Ny_target/2 : Ny_target/2-1) * sampling_cam_urad;

psf_export_path = 'C:\Users\ibrah\OneDrive\Bureau\stageLAM\Python_codes\psf_simulee1.mat';
save(psf_export_path, 'PSF_resampled', 'ax_resampled_urad', 'sampling_cam_urad');
fprintf('PSF exportée : %s\n', psf_export_path)