clear; close all; clc;

%% ========================================================
%  PHYSICAL SUPERPIXEL DMD TURBULENCE SIMULATOR
%
%  Nsp  et Ndmd sont INDÉPENDANTS.
%  Le LUT est construit pour des blocs n×n.
%  Chaque bloc DMD physique (nb×nb)
%  LUT par repmat+crop → pas d'interpolation, binaire exact.
%% ========================================================

%% ========================================================
% PARAMÈTRES
%% ========================================================

Nsp  = 50;      % nombre de superpixels par ligne
n    = 4;        % taille du superpixel LUT (2^(n²) états)
Ndmd = 400;      % résolution DMD physique — INDÉPENDANTE de Nsp

nb = floor(Ndmd / Nsp);   % taille d'un bloc DMD physique (pixels par superpixel)

if nb < 1
    error('Ndmd=%d trop petit pour Nsp=%d : nb=%d < 1', Ndmd, Nsp, nb);
end

fprintf('Nsp=%d  n=%d (LUT)  Ndmd=%d  nb=%d (bloc physique)\n', Nsp, n, Ndmd, nb);

r0 = 15;

%% ========================================================
% RAYON PINHOLE FOURIER
%% ========================================================

r = 0.065;

%% ========================================================
% PHASES DU SUPERPIXEL LUT  (n×n)
%% ========================================================

phases = reshape(2*pi*(0:n^2-1)/n^2, n, n);

%% ========================================================
% ÉCRAN DE PHASE KOLMOGOROV  (Nsp×Nsp)
%% ========================================================

fprintf('Generating Kolmogorov phase screen...\n')

fx = (-Nsp/2:Nsp/2-1) / Nsp;
[FX, FY] = meshgrid(fx, fx);
f = sqrt(FX.^2 + FY.^2);
f(f == 0) = 1e-6;

PSD = f.^(-11/3);

cn  = randn(Nsp) + 1i*randn(Nsp);
phi = real(ifft2(ifftshift(cn .* sqrt(PSD))));
phi = phi / std(phi(:));
phi = phi * pi;

%% ========================================================
% CHAMP COMPLEXE CIBLE
%% ========================================================

Etarget = exp(1i*phi);

%% ========================================================
% CONSTRUCTION DU LUT PHYSIQUE
%% ========================================================

fprintf('Building PHYSICAL LUT...\n')

Nstates = 2^(n^2);

Estate = zeros(Nstates, 1);
Bits   = false(Nstates, n^2);

%% --------------------------------------------------------
% Coordonnées Fourier pour un superpixel padé
%% --------------------------------------------------------

pad = 64;

fx_pad = (-pad/2:pad/2-1) / pad;
[FX_pad, FY_pad] = meshgrid(fx_pad, fx_pad);
FR_pad = sqrt(FX_pad.^2 + FY_pad.^2);

H = double(FR_pad < r);   % filtre pinhole

c0 = pad/2 - n/2 + 1;    % coin du bloc dans le pad

%% --------------------------------------------------------
% Boucle LUT
%% --------------------------------------------------------

for s = 0:Nstates-1

    bits = logical(bitget(s, 1:n^2));
    Bits(s+1, :) = bits;

    block = reshape(bits, n, n);

    complex_block = block .* exp(1i*phases);

    tmp = zeros(pad);
    tmp(c0:c0+n-1, c0:c0+n-1) = complex_block;

    F     = fftshift(fft2(tmp));
    Ffilt = F .* H;
    recon = ifft2(ifftshift(Ffilt));

    Estate(s+1) = recon(pad/2, pad/2);

end

Estate = Estate / max(abs(Estate));

%% ========================================================
% AFFICHAGE LUT
%% ========================================================

figure
plot(real(Estate), imag(Estate), '.')
axis equal; grid on
xlabel('Re(E)'); ylabel('Im(E)')
title(sprintf('Physical LUT  r = %.3f', r))

%% ========================================================
% RECONSTRUCTION DMD
%% ========================================================

fprintf('Reconstructing DMD...\n')

% Ndmd effectif = nb * Nsp  (on tronque si Ndmd n'est pas multiple de Nsp)
Ndmd_eff = nb * Nsp;

DMD           = zeros(Ndmd_eff, Ndmd_eff);
Ereconstructed = zeros(Nsp, Nsp);
idx_map        = zeros(Nsp, Nsp);

%% --------------------------------------------------------
% Pavage de la tuile LUT n×n → bloc nb×nb
%% --------------------------------------------------------

reps = ceil(nb / n);   % nombre de répétitions nécessaires

Rows = cell(Nsp, 1);

parfor iy = 1:Nsp

    Erow    = zeros(1, Nsp);
    DMDrow  = cell(1, Nsp);
    idx_row = zeros(1, Nsp);

    for ix = 1:Nsp

        target = Etarget(iy, ix);

        %% -------------------------------------------
        % état LUT le plus proche
        %% -------------------------------------------
        [~, idx] = min(abs(Estate - target));

        Erow(ix)    = Estate(idx);
        idx_row(ix) = idx;

        bits = Bits(idx, :);
        tile = reshape(bits, n, n);   % tuile n×n

        %% -------------------------------------------
        % pavage repmat + crop → bloc nb×nb binaire
        %% -------------------------------------------
        block_nb        = repmat(tile, reps, reps);
        block_nb        = block_nb(1:nb, 1:nb);
        DMDrow{ix}      = block_nb;

    end

    Ereconstructed(iy, :) = Erow;
    Rows{iy}              = DMDrow;
    idx_map(iy, :)        = idx_row;  %#ok<PFOUS>

end

%% ========================================================
% ASSEMBLAGE DMD COMPLET
%% ========================================================

for iy = 1:Nsp
    for ix = 1:Nsp
        ys = (iy-1)*nb + 1;
        xs = (ix-1)*nb + 1;
        DMD(ys:ys+nb-1, xs:xs+nb-1) = Rows{iy}{ix};
    end
end

%% ========================================================
% PROPAGATION OPTIQUE COMPLÈTE
%% ========================================================

fprintf('Applying full optical propagation...\n')

%% --------------------------------------------------------
% Phase LUT étendue au bloc nb×nb  (même pavage)
%% --------------------------------------------------------

ph_nb = repmat(phases, reps, reps);
ph_nb = ph_nb(1:nb, 1:nb);

ComplexDMD = zeros(Ndmd_eff, Ndmd_eff);

for iy = 1:Nsp
    for ix = 1:Nsp
        ys = (iy-1)*nb + 1;
        xs = (ix-1)*nb + 1;
        block = DMD(ys:ys+nb-1, xs:xs+nb-1);
        ComplexDMD(ys:ys+nb-1, xs:xs+nb-1) = block .* exp(1i*ph_nb);
    end
end

%% --------------------------------------------------------
% Pinhole physique sur la grille Ndmd_eff
%% --------------------------------------------------------

fx_full = (-Ndmd_eff/2:Ndmd_eff/2-1) / Ndmd_eff;
[FX_full, FY_full] = meshgrid(fx_full, fx_full);
FR_full = sqrt(FX_full.^2 + FY_full.^2);
Hfull   = double(FR_full < r);

%% --------------------------------------------------------
% Propagation
%% --------------------------------------------------------

F       = fftshift(fft2(ComplexDMD));
Ffilt   = F .* Hfull;
ReconFull = ifft2(ifftshift(Ffilt));

%% ========================================================
% EXTRACTION CHAMP PAR SUPERPIXEL
%% ========================================================

Efinal = zeros(Nsp, Nsp);

for iy = 1:Nsp
    for ix = 1:Nsp
        ys = (iy-1)*nb + 1;
        xs = (ix-1)*nb + 1;
        block = ReconFull(ys:ys+nb-1, xs:xs+nb-1);
        Efinal(iy, ix) = mean(block(:));
    end
end

%% ========================================================
% PHASES RECONSTRUITES
%% ========================================================

phi_rec = angle(Efinal);

%% ========================================================
% FIDÉLITÉ
%% ========================================================

Et = Etarget(:);
Er = Efinal(:);
Et = Et / norm(Et);
Er = Er / norm(Er);

Fidelity = abs(Et' * Er)^2;

fprintf('\n')
fprintf('=====================================\n')
fprintf('Nsp  = %d\n', Nsp)
fprintf('n    = %d  (LUT bloc)\n', n)
fprintf('Ndmd = %d  (physique)\n', Ndmd_eff)
fprintf('nb   = %d  (pixels/superpixel)\n', nb)
fprintf('r    = %.4f\n', r)
fprintf('Fidelity = %.5f\n', Fidelity)
fprintf('Error    = %.5f\n', 1-Fidelity)
fprintf('=====================================\n')

%% ========================================================
% AFFICHAGE PRINCIPAL
%% ========================================================

figure('Position',[100 100 1800 900])

subplot(2,3,1)
imagesc(phi); colorbar; axis image
title('Kolmogorov target phase (continue)')

subplot(2,3,2)
imagesc(angle(Etarget)); colorbar; axis image
title('Phase target wrapped [-\pi, \pi]')

subplot(2,3,3)
imagesc(abs(Etarget)); colorbar; axis image
title('Target amplitude')

subplot(2,3,4)
imagesc(DMD); colormap(gray); axis image
title(sprintf('%d×%d Superpixels,  %d×%d DMD  (bloc %d×%d)', ...
    Nsp, Nsp, Ndmd_eff, Ndmd_eff, nb, nb))

subplot(2,3,5)
imagesc(phi_rec); colorbar; axis image
title('Reconstructed phase')

subplot(2,3,6)
imagesc(angle(exp(1i*(phi - phi_rec)))); colorbar; axis image
title('Wrapped phase error')

sgtitle(sprintf('Physical Superpixel — Nsp=%d  Ndmd=%d  nb=%d  r=%.3f  Fidelity=%.4f', ...
    Nsp, Ndmd_eff, nb, r, Fidelity))

%% ========================================================
% COMPARAISON PSD
%% ========================================================

figure

PSD_target = abs(fftshift(fft2(phi))).^2;
PSD_rec    = abs(fftshift(fft2(phi_rec))).^2;

loglog(mean(PSD_target(:, Nsp/2:end), 1), 'LineWidth', 2); hold on
loglog(mean(PSD_rec(:,    Nsp/2:end), 1), 'LineWidth', 2);

grid on
xlabel('Spatial frequency'); ylabel('PSD')
legend('Target','Reconstructed')
title('Kolmogorov PSD comparison')