%% ========================================================
% SLM SUPERPIXEL — DIRECT PHASE ENCODING (v4 corrigé)
% Correction de l'erreur "Out of memory" et calcul correct de sampling_sim_urad
%% ========================================================

clear; close all; clc;

%% ========================================================
% PARAMÈTRES PHYSIQUES (À AJUSTER SELON VOTRE BANC)
%% ========================================================
Nsp    = 100;        % Nombre de superpixels par axe
n      = 4;          % Taille superpixel (n x n pixels SLM)
lambda = 633e-9;     % Longueur d'onde [m]
SLM_pitch = 9.2e-6;  % Pas du SLM [m]
f_L2 = 200e-3;       % Distance focale L2 [m]
f_L4bis = 200e-3;    % Distance focale L4bis [m] (pour conversion µrad)
grandissement = 100/200;  % f_L3/f_L2

fprintf('========================================\n');
fprintf('PARAMÈTRES PHYSIQUES:\n');
fprintf('Longueur d''onde       : %.1f nm\n', lambda*1e9);
fprintf('Pas SLM                : %.1f µm\n', SLM_pitch*1e6);
fprintf('Rayon pinhole          : 0.48 mm (fixe)\n');
fprintf('Grandissement          : %.2f\n', grandissement);
fprintf('========================================\n');

%% ========================================================
% PARAMÈTRES DE L'APERTURE (pinhole)
%% ========================================================
r_physique_mm = 0.48;  % Rayon physique du pinhole [mm] (mesuré expérimentalement)
r_normalized = 0.090;   % r normalisé pour MATLAB (0.090 cycles/pixel)

fprintf('r normalisé (MATLAB)   : %.6f\n', r_normalized);
fprintf('========================================\n');

%% ========================================================
% PHASE MAP DU SUPERPIXEL (4x4)
%% ========================================================
phases = reshape(2*pi*(0:n^2-1)/n^2, n, n); % Phases uniformément réparties

fprintf('Phase map (multiples de pi/8):\n');
disp(round(phases / (pi/8)));

%% ========================================================
% CIBLAGE (choisir un mode)
%% ========================================================
target_mode = 'kl';  % Options: 'turbulence', 'lg', 'image', 'kl'
mode_number = 10;    % Indice du mode KL (0-based)
gain_std = 1.0;      % Gain pour la variance du mode KL

%% --- Chargement du mode KL ---
modal_basis_file = 'C:\Users\ibrah\OneDrive\Bureau\stageLAM\Python_codes\utc_2026-04-28_09-37-25_KL_modal_basis.h5';
modal_basis = h5read(modal_basis_file, '/modal_basis');
modal_basis = permute(modal_basis, [3 2 1]);

KL_mode = squeeze(modal_basis(mode_number + 1, :, :));
KL_mode = imresize(KL_mode, [Nsp Nsp]);

% Application du gain sur la variance
pupil = KL_mode ~= 0;
KL_mode = KL_mode * gain_std;
KL_mode = KL_mode / max(abs(KL_mode(:))); % Normalisation

Etarget = exp(1i*KL_mode);
fprintf('Mode KL %d chargé et normalisé (gain_std=%.2f)\n', mode_number, gain_std);

%% ========================================================
% CONSTRUCTION DE LA LUT PHYSIQUE
%% ========================================================
fprintf('Construction de la LUT physique...\n');

K = 50000;  % Nombre d'états à tester
Estate = zeros(K, 1);
Bits = false(K, n^2);

pad = 128;
fx_lut = (-pad/2:pad/2-1) / pad;
[FX_lut, FY_lut] = meshgrid(fx_lut, fx_lut);
H = double(sqrt(FX_lut.^2 + FY_lut.^2) < r_normalized);

% Gestion des pixels OFF
use_random_off = true;
if use_random_off
    rng(50);
    phi_off_map = 2*pi * rand(n,n);
else
    phi_off_map = zeros(n,n);
end

off_block = exp(1i*phi_off_map);

for s = 1:K
    bits = rand(1, n^2) > 0.5;
    Bits(s,:) = bits;
    block = reshape(bits, n, n);

    complex_block = off_block;
    complex_block(block) = exp(1i*phases(block));

    tmp = zeros(pad);
    c0x = floor(pad/2 - n/2) + 1;
    c0y = floor(pad/2 - n/2) + 1;
    tmp(c0y:c0y+n-1, c0x:c0x+n-1) = complex_block;

    F = fftshift(fft2(tmp));
    Ffilt = F .* H;
    recon = ifft2(ifftshift(Ffilt));

    Estate(s) = recon(pad/2+1, pad/2+1);
end

Estate = Estate / max(abs(Estate));

%% ========================================================
% RECONSTRUCTION DU CHAMP CIBLE
%% ========================================================
fprintf('Reconstruction du champ cible...\n');

Ereconstructed = zeros(Nsp, Nsp);
for iy = 1:Nsp
    for ix = 1:Nsp
        target = Etarget(iy,ix);
        [~,idx] = min(abs(Estate - target));
        Ereconstructed(iy,ix) = Estate(idx);

        bits = Bits(idx,:);
        block = reshape(bits, n, n);

        ys = (iy-1)*n + 1;
        xs = (ix-1)*n + 1;

        tile = phi_off_map;
        tile(block) = phases(block);
        Device(ys:ys+n-1, xs:xs+n-1) = tile;
    end
end

%% ========================================================
% PROPAGATION COMPLÈTE (avec zero-padding) ET PSF
%% ========================================================
ComplexDevice = exp(1i*Device);
[XX, YY] = meshgrid((-size(ComplexDevice,2)/2:size(ComplexDevice,2)/2-1), ...
                   (-size(ComplexDevice,1)/2:size(ComplexDevice,1)/2-1));
pupil_mask = double(sqrt(XX.^2 + YY.^2) <= size(ComplexDevice,1)/2);
ComplexDevice = ComplexDevice .* pupil_mask;

pad_factor = 20;
Npad = size(ComplexDevice,1) * pad_factor;
ComplexDevice_pad = zeros(Npad);
offset = round((Npad - size(ComplexDevice,1)) / 2);
ComplexDevice_pad(offset+1:offset+size(ComplexDevice,1), offset+1:offset+size(ComplexDevice,2)) = ComplexDevice;

% Filtre de l'ouverture
fx_pad = (-Npad/2:Npad/2-1) / Npad;
[FX_pad, FY_pad] = meshgrid(fx_pad, fx_pad);
Hpad = double(sqrt(FX_pad.^2 + FY_pad.^2) < r_normalized);

% FFT et PSF
F_pad = fftshift(fft2(ComplexDevice_pad));
Ffilt = F_pad .* Hpad;
ReconFull_pad = ifft2(ifftshift(Ffilt));
PSF = abs(Ffilt).^2;
PSF = PSF / max(PSF(:));

%% ========================================================
% CALCUL CORRECT DE sampling_sim_urad (POUR PYTHON)
% Formule : sampling_sim_urad = (lambda * 1e6) / (SLM_pitch * grandissement)
%% ========================================================
sampling_sim_urad = (lambda * 1e6) / (SLM_pitch * grandissement);  % µrad/pixel
sampling_meas_urad = (3.45e-6 / f_L4bis) * 1e6;  % µrad/pixel (caméra)

fprintf('sampling_sim  = %.4f µrad/pixel\n', sampling_sim_urad);
fprintf('sampling_meas = %.4f µrad/pixel\n', sampling_meas_urad);
fprintf('Facteur de rééchantillonnage = %.2f\n', sampling_sim_urad / sampling_meas_urad);

%% ========================================================
% EXPORT DE LA PSF POUR PYTHON (avec taille raisonnable)
%% ========================================================
% Limiter la taille maximale pour éviter les problèmes mémoire
MAX_PSF_SIZE = 512;  % Taille maximale en pixels

Ny_psf = size(PSF, 1);
Nx_psf = size(PSF, 2);

% Calcul du facteur de rééchantillonnage
factor = sampling_sim_urad / sampling_meas_urad;

% Limiter la taille de la PSF rééchantillonnée
Ny_target = min(MAX_PSF_SIZE, max(1, round(Ny_psf * factor)));
Nx_target = min(MAX_PSF_SIZE, max(1, round(Nx_psf * factor)));

fprintf('PSF originale size      : %d x %d\n', Ny_psf, Nx_psf);
fprintf('PSF rééchantillonnée size : %d x %d (limité à %d)\n', Ny_target, Nx_target, MAX_PSF_SIZE);

% Rééchantillonnage avec vérification de la taille
if Ny_target > MAX_PSF_SIZE || Nx_target > MAX_PSF_SIZE
    warning('Taille de PSF trop grande, limitation appliquée');
end

PSF_resampled = imresize(PSF, [Ny_target Nx_target], 'bicubic');
PSF_resampled = PSF_resampled / max(PSF_resampled(:));

% Calcul de l'axe en µrad
ax_resampled_urad = (-Ny_target/2 : Ny_target/2-1) * sampling_meas_urad;

psf_export_path = 'C:\Users\ibrah\OneDrive\Bureau\stageLAM\Python_codes\psf_simulee1.mat';
save(psf_export_path, 'PSF_resampled', 'ax_resampled_urad', 'sampling_sim_urad');
fprintf('PSF exportée pour Python : %s\n', psf_export_path);

%% ========================================================
% AFFICHAGE DES RÉSULTATS
%% ========================================================
figure('Position', [100 100 1800 900]);

subplot(2,4,1);
imagesc(abs(Etarget)); axis image; colorbar; colormap(gray);
title('Amplitude cible');

subplot(2,4,2);
imagesc(angle(Etarget)); axis image; colorbar; colormap(hsv);
title('Phase cible');

subplot(2,4,3);
imagesc(abs(Ereconstructed)); axis image; colorbar; colormap(gray);
title('Amplitude reconstruite');

subplot(2,4,4);
imagesc(angle(Ereconstructed)); axis image; colorbar; colormap(hsv);
title('Phase reconstruite');

subplot(2,4,5);
imagesc(abs(Etarget) - abs(Ereconstructed)/max(abs(Ereconstructed(:)))*max(abs(Etarget(:))));
axis image; colorbar; colormap(gray);
title('Erreur amplitude');

subplot(2,4,6);
imagesc(angle(exp(1i*(angle(Etarget) - angle(Ereconstructed)))));
axis image; colorbar; colormap(hsv);
title('Erreur phase');

subplot(2,4,7);
imagesc(Device); colormap(hsv);
axis image; colorbar;
title('Carte de phase SLM');

subplot(2,4,8);
plot(real(Estate), imag(Estate), '.', 'MarkerSize', 2);
hold on;
plot(real(Etarget(:)), imag(Etarget(:)), 'r.', 'MarkerSize', 3);
axis equal; grid on;
xlabel('Re(E)'); ylabel('Im(E)');
legend('LUT atteignable', 'Points cibles');
title('Plan complexe');

fidelity = abs(sum(conj(Etarget(:)) .* Ereconstructed(:)))^2 / ...
           (sum(abs(Etarget(:)).^2) * sum(abs(Ereconstructed(:)).^2));
sgtitle(sprintf('Reconstruction SLM [KL %d] — Fidelity = %.4f', mode_number, fidelity));

%% ========================================================
% EXPORT DE LA CARTE SLM POUR LE DISPOSITIF
%% ========================================================
SLM_map = uint8(round(mod(Device, 2*pi) / (2*pi) * 255));

figure('Position', [100 100 700 700]);
imagesc(SLM_map, [0 255]);
colormap(gray);
colorbar;
axis image;
title(sprintf('Carte SLM — %d x %d pixels — 8 bit', size(SLM_map,2), size(SLM_map,1)));

%% ========================================================
% AFFICHAGE DE LA PSF (LOG SCALE)
%% ========================================================
figure('Position', [100 100 1200 500]);

subplot(1,2,1);
imagesc(ax_resampled_urad, ax_resampled_urad, log10(PSF_resampled + 1e-6));
axis image; colorbar; colormap(hot);
title('PSF (log10) — µrad');
xlabel('angle [µrad]'); ylabel('angle [µrad]');
xlim([-1500 1500]); ylim([-1500 1500]);

subplot(1,2,2);
imagesc(fx_pad, fx_pad, log10(PSF + 1e-6));
axis image; colorbar; colormap(hot);
title('PSF (log10) — cycles/pixel');
xlabel('f_x'); ylabel('f_y');
xlim([-5*r_normalized 5*r_normalized]); ylim([-5*r_normalized 5*r_normalized])
sgtitle(sprintf('PSF simulée — n=%d — Nsp=%d — r=%.4f', n, Nsp, r_normalized));