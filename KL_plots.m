%% ========================================================
% SWEEP : Fidelity vs r, pour différents modes KL et Nsp
%% ========================================================

clear; close all; clc;

%% --- Paramètres fixes ---
n      = 4;
lambda = 633e-9;
SLM_pitch = 9.2e-6;
use_random_off = true;
pad    = 128;

modal_basis_file = 'C:\Users\ibrah\OneDrive\Bureau\stageLAM\Python_codes\utc_2026-04-28_09-37-25_KL_modal_basis.h5';
modal_basis = h5read(modal_basis_file, '/modal_basis');
modal_basis = permute(modal_basis, [3 2 1]);

%% --- Axes du sweep ---
r_values   = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12];
KL_modes   = [10, 20, 50, 150, 250, 350, 360, 370, 380, 390, 400, 410 ];
Nsp_values = [100];

%% --- Stockage des résultats ---
% Fidelity(i_r, i_kl, i_nsp)
Fidelity = zeros(length(r_values), length(KL_modes), length(Nsp_values));

%% --- Boucle principale ---
total = length(r_values) * length(KL_modes) * length(Nsp_values);
count = 0;

for i_nsp = 1:length(Nsp_values)
    Nsp = Nsp_values(i_nsp);

    for i_kl = 1:length(KL_modes)
        mode_number = KL_modes(i_kl);

        %% Préparer le mode KL (une fois par mode+Nsp)
        KL_mode = squeeze(modal_basis(mode_number + 1, :, :));
        KL_mode = imresize(KL_mode, [Nsp Nsp]);
        KL_mode = KL_mode / max(abs(KL_mode(:)));
        Etarget = abs(KL_mode) .* exp(1i * pi * (KL_mode < 0));

        for i_r = 1:length(r_values)
            r = r_values(i_r);
            count = count + 1;
            fprintf('[%d/%d]  Nsp=%d  KL=%d  r=%.3f\n', count, total, Nsp, mode_number, r);

            %% --- OFF pixel map ---
            if use_random_off
                rng(50)   % graine fixe, le chiffre n'a pas d'importance, ligne unique au mode KL, ZK
                phi_off_map = 2*pi * rand(n,n);
            else
                phi_off_map = zeros(n,n);
            end

            %% --- Phase map du superpixel ---
            phases = reshape(2*pi*(0:n^2-1)/n^2, n, n);

            %% --- LUT ---
            Nstates = 2^(n^2);
            Estate  = zeros(Nstates, 1);
            Bits    = false(Nstates, n^2);

            fx_lut = (-pad/2:pad/2-1) / pad;
            [FX_lut, FY_lut] = meshgrid(fx_lut, fx_lut);
            H = double(sqrt(FX_lut.^2 + FY_lut.^2) < r);

            off_block = exp(1i*phi_off_map);

            for s = 0:Nstates-1
                bits        = logical(bitget(s, 1:n^2));
                Bits(s+1,:) = bits;
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
                Estate(s+1) = recon(pad/2+1, pad/2+1);
            end

            Estate = Estate / max(abs(Estate));

            %% --- Reconstruction ---
            Ereconstructed = zeros(Nsp, Nsp);
            for iy = 1:Nsp
                for ix = 1:Nsp
                    target  = Etarget(iy,ix);
                    [~,idx] = min(abs(Estate - target));
                    Ereconstructed(iy,ix) = Estate(idx);
                end
            end

            %% --- Fidelité ---
            Et = Etarget(:);  Et = Et / norm(Et);
            Er = Ereconstructed(:); Er = Er / norm(Er);
            Fidelity(i_r, i_kl, i_nsp) = abs(Et'*Er)^2;

        end % r
    end % KL
end % Nsp

% Calculate and display the fidelity results

save('fidelity_results_rng50_Nsp100_n4.mat', 'Fidelity', 'r_values', 'KL_modes', 'Nsp_values', 'n', 'use_random_off')

%% ========================================================
% FIGURE : Fidelity vs r  (1 subplot par mode KL, 1 courbe par Nsp)
%% ========================================================

colors    = lines(length(Nsp_values));
markers   = {'o', 's', '^'};

figure('Position', [100 100 1400 450])

for i_kl = 1:length(KL_modes)
    subplot(1, 3, i_kl)
    hold on; grid on; box on

    for i_nsp = 1:length(Nsp_values)
        plot(r_values, Fidelity(:, i_kl, i_nsp), ...
            '-', ...
            'Marker',    markers{i_nsp}, ...
            'Color',     colors(i_nsp,:), ...
            'LineWidth', 1.8, ...
            'MarkerSize', 7, ...
            'DisplayName', sprintf('N_{sp} = %d', Nsp_values(i_nsp)))
    end

    xlabel('r  [norm. freq.]', 'FontSize', 11)
    ylabel('Fidelity',         'FontSize', 11)
    title(sprintf('KL mode %d', KL_modes(i_kl)), 'FontSize', 13)
    legend('Location', 'best', 'FontSize', 10)
    ylim([max(0.8, min(Fidelity(:,i_kl,:), [], 'all') - 0.01)  1])
    xlim([r_values(1) r_values(end)])
end

sgtitle(sprintf('Fidelity vs r  —  n=%d superpixels, phi_{off}=%s', ...
    n, ternary_str(use_random_off, 'random', '0')), 'FontSize', 14)

%% ========================================================
function s = ternary_str(cond, a, b)
    if cond, s = a; else, s = b; end
end
%%-----------repairing...
Nsp = 25;
KL_mode = squeeze(modal_basis(11, :, :));
KL_mode = imresize(KL_mode, [Nsp Nsp]);
KL_mode = KL_mode / max(abs(KL_mode(:)));
Etarget = abs(KL_mode) .* exp(1i * pi * (KL_mode < 0));

Ereconstructed = zeros(Nsp, Nsp);
for iy = 1:Nsp
    for ix = 1:Nsp
        target = Etarget(iy,ix);
        [~,idx] = min(abs(Estate - target));
        Ereconstructed(iy,ix) = Estate(idx);
    end
end

Et = Etarget(:); Et = Et / norm(Et);
Er = Ereconstructed(:); Er = Er / norm(Er);
F_test = abs(Et'*Er)^2;
disp(F_test)