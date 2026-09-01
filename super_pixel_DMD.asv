%% ============================================================
% super_pixel_DMD.m  —  Goorden 2014, vrai DMD ±12°
%
% Modèle physique :
%   Miroirs ±θ=±12°, éclairage Littrow à 2θ=24°.
%   ON  (+12°) → 0° → tache spéculaire ON = origine plan focal.
%   OFF (−12°) → −48° → ~148 mm off-axis (filtré).
%
% Modèle de simulation :
%   ON  pixel → amplitude 1  (phase 0 dans le plan miroir)
%   OFF pixel → porteur proxy à (f_off, f_off) = (0.40, 0.40) cy/miroir
%               Pour Ndmd=400 : (f_off−f0x)*Ndmd et (f_off−f0y)*Ndmd sont
%               entiers → contribution OFF à l'ouverture = EXACTEMENT 0 ✓
%
%   Ouverture à (f0x, f0y) = (−1/n², +1/n) cy/miroir [position Goorden].
%
% LUT :
%   Coefficient DFT direct à (f0x, f0y) :
%     Estate(s) = Σ_{ON (ir,ic)} exp(−i·2π·(f0x·ic + f0y·ir))
%   OFF ignoré (contribution = 0 en propagation complète).
%
% Extraction champ reconstruit :
%   1. IFFT(FFT(Device) × H_aperture)
%   2. Démodulation : × exp(−i·2π·(f0x·IC + f0y·IR))
%   3. Moyenne sur chaque bloc n×n
%   4. Correction porteur colonne : × exp(−i·2π·(ix−1)/n)
%
% Le signal utile apparaît à la position physique Goorden dans le plan focal.
%% ============================================================

clear; close all; clc;

%% ============================================================
%% PARAMÈTRES
%% ============================================================

Nsp    = 100;
n      = 4;
Ndmd   = Nsp * n;       % = 400

lambda = 633e-9;
d      = 13.68e-6;
theta  = 12 * pi/180;

f_L2   = 300e-3;

target_mode = 'focus';   % 'turbulence' | 'LG' | 'tiptilt' | 'focus' | 'flat'

%% Aberrations additionnelles (s'ajoutent à la phase de base)
add_tiptilt  = false;   % true = ajouter tip/tilt
tip_rad      = 2.0;     % amplitude tip  [rad peak-to-edge, axe x]
tilt_rad     = 3.1;     % amplitude tilt [rad peak-to-edge, axe y]

add_focus    = true;   % true = ajouter focus (défocalisation)
focus_rad    = 1.0;     % amplitude focus [rad peak-to-edge]

%% Amplitude cible [0 → 1]
%   Pour cibles phase-pure : garder 1.0 (les points LUT du bord donnent
%   la meilleure couverture angulaire, et la fidélité est invariante après norm.)
%   Utile seulement pour cibles amplitude+phase (ex. LG) : réduire < 1.
amp_target = 1.0;

%% ============================================================
%% GÉOMÉTRIE PHYSIQUE
%% ============================================================

theta_illum       = 2*theta;
theta_OFF_reflect = (2*(-theta) - theta_illum);   % −48°
x_OFF_phys        = f_L2 * sin(theta_OFF_reflect);  % ≈ −148 mm

%% ============================================================
%% OUVERTURE GOORDEN
%% ============================================================

% Fréquences normalisées [cy/miroir] -------------------------//////////?
f0x_norm = -1/n^2;    % = −0.0625
f0y_norm = +1/n;      % = +0.25

% Position physique dans le plan focal [m]
x_ap = f0x_norm * lambda * f_L2 / d;   % ≈ −0.579 mm
y_ap = f0y_norm * lambda * f_L2 / d;   %  +2.316 mm

% Rayon [cy/miroir] — critère Nyquist : r < 1/(2n) = 0.125
r_norm = 0.10;
r_phys = r_norm * lambda * f_L2 / d;   % [m]

fprintf('=== DMD ±%.0f°  (Littrow %.0f°) ===\n', theta*180/pi, theta_illum*180/pi)
fprintf('Tache ON         : (0, 0) mm\n')
fprintf('Tache OFF réelle : (%.1f, 0) mm  [filtrée]\n', x_OFF_phys*1e3)
fprintf('\nPinhole sur le banc (plan focal L2, f=%.0f mm) :\n', f_L2*1e3)
fprintf('  Centre : (%.3f, %.3f) mm\n', x_ap*1e3, y_ap*1e3)
fprintf('  Rayon  : %.3f mm  (diam. %.3f mm)\n', r_phys*1e3, 2*r_phys*1e3)
fprintf('  r norm : %.4f  < 1/(2n)=%.4f  → %s\n\n', ...
        r_norm, 1/(2*n), ternary_str(r_norm < 1/(2*n),'OK ✓','FAIL ✗'))

%% ============================================================
%% PORTEUR MIROIR OFF  (proxy simulation)
%%
%%   f_off = 0.40 cy/miroir
%%   (f_off − f0x) × 400 = (0.40+0.0625)×400 = 185  ← entier
%%   (f_off − f0y) × 400 = (0.40−0.25)×400   =  60  ← entier
%%   → contribution OFF à l'ouverture = 0 EXACTEMENT pour Ndmd=400
%% ============================================================

f_off = 0.40;

[IC_g, IR_g]  = meshgrid(0:Ndmd-1, 0:Ndmd-1);
phi_off_g     = 2*pi*(f_off*IC_g + f_off*IR_g);   % porteur global OFF

[ic_sp, ir_sp] = meshgrid(0:n-1, 0:n-1);
phi_off_sp     = 2*pi*(f_off*ic_sp + f_off*ir_sp); % porteur local OFF

%% Noyau DFT à (f0x, f0y)
demod_kernel = exp(-1i*2*pi*(f0x_norm*ic_sp + f0y_norm*ir_sp));  % n×n

%% ============================================================
%% CIBLE
%% ============================================================

%% Grille normalisée commune [-1, +1]
[Xn, Yn] = meshgrid(linspace(-1,1,Nsp), linspace(-1,1,Nsp));
Rn = sqrt(Xn.^2 + Yn.^2);

switch lower(target_mode)
    case 'turbulence'
        fprintf('Cible : turbulence de Kolmogorov\n')
        fx_t = (-Nsp/2:Nsp/2-1)/Nsp;
        [FXt,FYt] = meshgrid(fx_t,fx_t);
        ft = sqrt(FXt.^2+FYt.^2); ft(ft==0)=1e-6;
        PSD = ft.^(-11/3);
        cn  = randn(Nsp)+1i*randn(Nsp);
        phi = real(ifft2(ifftshift(cn.*sqrt(PSD))));
        phi = phi/std(phi(:))*pi;

    case 'tiptilt'
        fprintf('Cible : tip/tilt pur\n')
        phi = zeros(Nsp);

    case 'focus'
        fprintf('Cible : focus pur\n')
        phi = zeros(Nsp);

    case 'flat'
        fprintf('Cible : phase nulle (champ uniforme)\n')
        phi = zeros(Nsp);

    case 'lg'
        fprintf('Cible : LG_1^0\n')
        PHI = atan2(Yn, Xn);
        A   = Rn.*exp(-Rn.^2);
        Etarget = A/max(A(:)).*exp(1i*PHI);
        phi = angle(Etarget);

    otherwise
        error('target_mode inconnu : ''%s''\nModes valides : turbulence | tiptilt | focus | flat | LG', ...
              target_mode)
end

%% Aberrations additionnelles superposées à phi
phi_aberr = zeros(Nsp);

if add_tiptilt
    % Tip (x) + Tilt (y) : rampes linéaires, amplitude peak-to-edge
    phi_aberr = phi_aberr + tip_rad * Xn + tilt_rad * Yn;
    fprintf('  + Tip  = %.2f rad   Tilt = %.2f rad\n', tip_rad, tilt_rad)
end

if add_focus
    % Focus (défocalisation) : paraboloïde, 0 au bord → focus_rad au centre
    phi_aberr = phi_aberr + focus_rad * (1 - Rn.^2);
    fprintf('  + Focus = %.2f rad (pic au centre)\n', focus_rad)
end

phi = phi + phi_aberr;

%% Champ cible complexe
if ~exist('Etarget','var')
    Etarget = exp(1i*phi);
end

%% ============================================================
%% BUILD LUT
%%
%% Estate(s) = Σ_{ON (ir,ic)} exp(−i·2π·(f0x·ic + f0y·ir))
%%
%% Calcul direct sans FFT : produit scalaire du masque ON
%% avec le noyau DFT à la fréquence aperture.
%% OFF ignoré (contribution nulle à l'ouverture dans la
%% propagation complète pour Ndmd=400 et f_off=0.40).
%% ============================================================

fprintf('LUT (%d états) …\n', 2^(n^2))

Nstates = 2^(n^2);
Estate  = zeros(Nstates, 1);
Bits    = false(Nstates, n^2);

for s = 0:Nstates-1
    bits        = logical(bitget(s, 1:n^2));
    Bits(s+1,:) = bits;
    block       = reshape(bits, n, n);
    % Coefficient DFT direct à (f0x, f0y) — seuls les ON contribuent
    Estate(s+1) = sum(sum(double(block) .* demod_kernel));
end

Estate = Estate / max(abs(Estate));

%% Amplitude optimale : met le cercle cible dans la zone dense de la LUT
if ischar(amp_target) && strcmpi(amp_target, 'auto')
    A_opt = median(abs(Estate));
else
    A_opt = amp_target;
end
fprintf('Amplitude cible  : %.4f  (médiane LUT = %.4f)\n', A_opt, median(abs(Estate)))

%% Rescaler Etarget à l'amplitude optimale (phase inchangée)
Etarget = A_opt * exp(1i*angle(Etarget));

figure('Name','LUT')
plot(real(Estate), imag(Estate), '.', 'MarkerSize', 2); hold on
th_lut = linspace(0,2*pi,300);
plot(A_opt*cos(th_lut), A_opt*sin(th_lut), 'r-', 'LineWidth', 1.5)
hold off; axis equal; grid on; xlabel('Re(E)'); ylabel('Im(E)')
legend('LUT', sprintf('Cible A=%.3f', A_opt), 'Location','best')
title(sprintf('DMD LUT — n=%d, r=%.3f cy/miroir', n, r_norm))

%% ============================================================
%% BUILD DMD DEVICE + RECONSTRUCTION
%% ============================================================

fprintf('Device et reconstruction …\n')

ComplexDevice = exp(1i*phi_off_g);   % tout initialisé à OFF
BinaryDevice  = zeros(Ndmd, Ndmd);
Ereconstructed = zeros(Nsp, Nsp);

for iy = 1:Nsp
    for ix = 1:Nsp
        target = Etarget(iy,ix);
        [~, idx] = min(abs(Estate - target));
        Ereconstructed(iy,ix) = Estate(idx);

        bits  = Bits(idx,:);
        block = reshape(bits, n, n);

        ys = (iy-1)*n + 1;
        xs = (ix-1)*n + 1;

        tile = exp(1i*phi_off_sp);   % OFF
        tile(block) = 1;             % ON = amplitude 1

        ComplexDevice(ys:ys+n-1, xs:xs+n-1) = tile;
        BinaryDevice(ys:ys+n-1, xs:xs+n-1)  = double(block);
    end
end

%% ============================================================
%% PROPAGATION FOURIER COMPLÈTE
%% ============================================================

fprintf('Propagation Fourier …\n')

%% ---- FFT sans padding  →  reconstruction du champ ----
Frecon = fftshift(fft2(ComplexDevice));

fx_r = (-Ndmd/2:Ndmd/2-1) / Ndmd;
[FXr, FYr] = meshgrid(fx_r, fx_r);
Hr    = double((FXr - f0x_norm).^2 + (FYr - f0y_norm).^2 < r_norm^2);
Ffilt     = Frecon .* Hr;
ReconFull = ifft2(ifftshift(Ffilt));

%% ---- FFT zero-paddée  →  affichage plan focal haute résolution ----
pad_factor = 8;           % 8× → ~8 pixels par résolution diffraction
Npad       = pad_factor * Ndmd;

DevicePad = zeros(Npad, Npad);
off_p     = (Npad - Ndmd) / 2;
DevicePad(off_p+1:off_p+Ndmd, off_p+1:off_p+Ndmd) = ComplexDevice;

Ffull = fftshift(fft2(DevicePad));

%% Axes physiques plan focal [mm]
dx_mm = lambda * f_L2 / (Npad * d) * 1e3;
x_ax  = (-Npad/2:Npad/2-1) * dx_mm;

%% Filtre d'ouverture dans la grille paddée
%%  Le zero-padding affine l'échantillonnage mais NE change PAS la plage :
%%  fx_pad est toujours en cy/miroir, de -0.5 à +0.5.
fx_pad = (-Npad/2:Npad/2-1) / Npad;   % [cy/miroir]
[FXp, FYp] = meshgrid(fx_pad, fx_pad);
Hfull = double((FXp - f0x_norm).^2 + (FYp - f0y_norm).^2 < r_norm^2);
Ffilt_disp = Ffull .* Hfull;

%% ============================================================
%% AFFICHAGE PLAN FOCAL
%% ============================================================

Imax    = max(abs(Ffull(:)))^2;
Ilog_no = 10*log10(abs(Ffull).^2      / Imax + 1e-8);
Ilog_pi = 10*log10(abs(Ffilt_disp).^2 / Imax + 1e-8);

th_c  = linspace(0, 2*pi, 300);
xc_mm = x_ap*1e3 + r_phys*1e3*cos(th_c);
yc_mm = y_ap*1e3 + r_phys*1e3*sin(th_c);

x_off_sim_mm = f_off * lambda * f_L2 / d * 1e3;   % position proxy OFF

%% ---- SANS pinhole ----
figure('Name','Plan focal — SANS pinhole','Position',[30 60 1200 560])
imagesc(x_ax, x_ax, Ilog_no, [-60 0])
colormap(hot); c=colorbar; c.Label.String='dB'; axis image xy
xlabel('x [mm]'); ylabel('y [mm]')
title(sprintf('Plan focal — SANS pinhole   (λ=%dnm, f=%.0fmm, d=%.2fµm)', ...
    round(lambda*1e9), f_L2*1e3, d*1e6))
hold on
plot(0, 0, 'g+', 'MarkerSize',16, 'LineWidth', 2.5)
text(0.10, -0.22, 'Tache ON (0,0)', 'Color','g', 'FontWeight','bold', 'FontSize',9)
plot(xc_mm, yc_mm, 'c-', 'LineWidth', 2.5)
plot(x_ap*1e3, y_ap*1e3, 'c+', 'MarkerSize',12, 'LineWidth', 2.5)
text(x_ap*1e3+0.15, y_ap*1e3+0.12, ...
    sprintf('Pinhole Goorden\n(%.2f, %.2f) mm\nr = %.2f mm', ...
    x_ap*1e3, y_ap*1e3, r_phys*1e3), ...
    'Color','c', 'FontSize',9, 'BackgroundColor','k')
plot(x_off_sim_mm, x_off_sim_mm, 'r^', 'MarkerSize',9, 'LineWidth',1.5)
text(x_off_sim_mm+0.12, x_off_sim_mm, ...
    sprintf('OFF proxy\n[réel: %.0fmm]', abs(x_OFF_phys*1e3)), ...
    'Color','r', 'FontSize',8)
hold off

%% ---- AVEC pinhole ----
figure('Name','Plan focal — AVEC pinhole','Position',[50 80 1200 560])
imagesc(x_ax, x_ax, Ilog_pi, [-60 0])
colormap(hot); c=colorbar; c.Label.String='dB'; axis image xy
xlabel('x [mm]'); ylabel('y [mm]')
title(sprintf('Plan focal — AVEC pinhole   (r = %.3f cy/miroir = %.3f mm)', ...
    r_norm, r_phys*1e3))
hold on
plot(xc_mm, yc_mm, 'c-', 'LineWidth', 2.5)
plot(x_ap*1e3, y_ap*1e3, 'c+', 'MarkerSize',12, 'LineWidth', 2.5)
text(x_ap*1e3+0.15, y_ap*1e3+0.12, ...
    sprintf('Pinhole\n(%.2f, %.2f) mm', x_ap*1e3, y_ap*1e3), ...
    'Color','c', 'FontSize',9)
plot(0, 0, 'g+', 'MarkerSize',12, 'LineWidth', 2)
hold off

%% ---- Zoom ouverture ----
zoom_h = r_phys*1e3 * 8;

figure('Name','Zoom : région pinhole','Position',[50 50 1000 450])
subplot(1,2,1)
imagesc(x_ax, x_ax, Ilog_no, [-60 0])
colormap(hot); colorbar; axis image xy
xlim(x_ap*1e3 + [-zoom_h zoom_h]); ylim(y_ap*1e3 + [-zoom_h zoom_h])
xlabel('x [mm]'); ylabel('y [mm]'); title('Zoom — sans pinhole')
hold on; plot(xc_mm,yc_mm,'c-','LineWidth',2); hold off

subplot(1,2,2)
imagesc(x_ax, x_ax, Ilog_pi, [-60 0])
colormap(hot); colorbar; axis image xy
xlim(x_ap*1e3 + [-zoom_h zoom_h]); ylim(y_ap*1e3 + [-zoom_h zoom_h])
xlabel('x [mm]'); ylabel('y [mm]'); title('Zoom — avec pinhole')
hold on; plot(xc_mm,yc_mm,'c-','LineWidth',2); hold off

%% ============================================================
%% EXTRACTION DU CHAMP RECONSTRUIT
%%
%% Après filtrage off-axis à (f0x,f0y), ReconFull porte le
%% porteur spatial exp(+i·2π·(f0x·IC + f0y·IR)).
%% 1. Démodulation → champ lentement variable
%% 2. Moyenne par bloc n×n
%% 3. Correction porteur de colonne × exp(−i·2π·(ix−1)/n)
%% ============================================================

[IC_d, IR_d] = meshgrid(0:Ndmd-1, 0:Ndmd-1);
ReconDemod   = ReconFull .* exp(-1i*2*pi*(f0x_norm*IC_d + f0y_norm*IR_d));

Efinal = zeros(Nsp, Nsp);
for iy = 1:Nsp
    for ix = 1:Nsp
        ys  = (iy-1)*n + 1;
        xs  = (ix-1)*n + 1;
        blk = ReconDemod(ys:ys+n-1, xs:xs+n-1);
        Efinal(iy,ix) = mean(blk(:));
    end
end

% Correction porteur de colonne :
% Efinal(iy,ix) ≈ exp(+i·2π·(ix−1)/n) × Estate(best_s)
% → multiplier par exp(−i·2π·(ix−1)/n)
[IX, ~] = meshgrid(1:Nsp, 1:Nsp);
Efinal  = Efinal .* exp(-1i*2*pi*(IX-1)/n);

phi_rec = angle(Efinal);

%% ============================================================
%% FIDÉLITÉ
%% ============================================================

Et = Etarget(:); Et = Et/norm(Et);
Er = Efinal(:);  Er = Er/norm(Er);
Fidelity = abs(Et'*Er)^2;

fprintf('\n=====================================\n')
fprintf('n         = %d  (%d miroirs/superpixel)\n', n, n^2)
fprintf('r         = %.4f  < 1/(2n)=%.4f\n', r_norm, 1/(2*n))
fprintf('Fidélité  = %.5f\n', Fidelity)
fprintf('=====================================\n')

%% ============================================================
%% RÉSULTATS
%% ============================================================

figure('Name','Reconstruction','Position',[80 80 1600 800])

subplot(2,4,1); imagesc(abs(Etarget));  axis image; colorbar; colormap(gray)
title('Amplitude cible')
subplot(2,4,2); imagesc(angle(Etarget)); axis image; colorbar; colormap(hsv)
title('Phase cible')
subplot(2,4,3); imagesc(abs(Efinal));  axis image; colorbar; colormap(gray)
title('Amplitude reconstruite')
subplot(2,4,4); imagesc(phi_rec); axis image; colorbar; colormap(hsv)
title('Phase reconstruite')

subplot(2,4,5)
imagesc(BinaryDevice); colormap(gray); axis image; colorbar
title(sprintf('Masque DMD (%d×%d)', Ndmd, Ndmd))

subplot(2,4,6)
imagesc(angle(exp(1i*(angle(Etarget)-phi_rec)))); axis image; colorbar; colormap(hsv)
title('Erreur de phase (enroulée)')

subplot(2,4,7)
ref = abs(Efinal)/max(abs(Efinal(:)))*max(abs(Etarget(:)));
imagesc(abs(Etarget)-ref); axis image; colorbar
title('Erreur d''amplitude')

subplot(2,4,8)
plot(real(Estate), imag(Estate), '.', 'MarkerSize',2); hold on
plot(real(Etarget(:)), imag(Etarget(:)), 'r.', 'MarkerSize',3)
axis equal; grid on; xlabel('Re(E)'); ylabel('Im(E)')
legend('LUT','Cible','Location','best')
title('Plan complexe')

sgtitle(sprintf('DMD superpixel [%s] — n=%d — θ=%.0f° — r=%.3f — Fidélité=%.4f', ...
    target_mode, n, theta*180/pi, r_norm, Fidelity))

%% ============================================================
%% PSF — plan image (TF du champ pupille)
%% ============================================================

pad_psf = 4;
Npsf    = pad_psf * Nsp;

Et_pad  = zeros(Npsf, Npsf);
Er_pad  = zeros(Npsf, Npsf);
off_psf = (Npsf - Nsp) / 2;
Et_pad(off_psf+1:off_psf+Nsp, off_psf+1:off_psf+Nsp) = Etarget;
Er_pad(off_psf+1:off_psf+Nsp, off_psf+1:off_psf+Nsp) = Efinal;

PSF_t = abs(fftshift(fft2(Et_pad))).^2;  PSF_t = PSF_t / max(PSF_t(:));
PSF_r = abs(fftshift(fft2(Er_pad))).^2;  PSF_r = PSF_r / max(PSF_r(:));

% Axes angulaires [λ/D] où D = Nsp * (d*n) = taille du champ de miroirs
dx_psf = 1 / (pad_psf * Nsp);   % [cy/superpixel] par bin
u_ax   = (-Npsf/2:Npsf/2-1) * dx_psf * Nsp;   % [λ/D]

% Zoom sur le cœur de la PSF (±8 λ/D)
zoom_psf = 8;

figure('Name','PSF reconstruite','Position',[100 100 1000 480])
subplot(1,2,1)
imagesc(u_ax, u_ax, 10*log10(PSF_t + 1e-6), [-40 0])
colormap(hot); colorbar; axis image xy
xlim([-zoom_psf zoom_psf]); ylim([-zoom_psf zoom_psf])
xlabel('\lambda/D'); ylabel('\lambda/D')
title('PSF — cible')

subplot(1,2,2)
imagesc(u_ax, u_ax, 10*log10(PSF_r + 1e-6), [-40 0])
colormap(hot); colorbar; axis image xy
xlim([-zoom_psf zoom_psf]); ylim([-zoom_psf zoom_psf])
xlabel('\lambda/D'); ylabel('\lambda/D')
title('PSF — reconstruite (DMD)')

sgtitle(sprintf('PSF [%s] — Fidélité=%.4f', target_mode, Fidelity))

%% Zoom masque
show = min(40,Nsp); npx = show*n;
figure('Name','Masque DMD','Position',[50 80 700 700])
imagesc(BinaryDevice(1:npx,1:npx)); colormap(gray); axis image; colorbar
title(sprintf('Masque DMD — %d×%d superpixels', show, show))
hold on
for k=0:show
    xline(k*n+0.5,'b-','LineWidth',0.4,'Alpha',0.3)
    yline(k*n+0.5,'b-','LineWidth',0.4,'Alpha',0.3)
end
hold off

%% ============================================================
%% SAUVEGARDE MASQUE BINAIRE
%% ============================================================

dmd_mask   = uint8(BinaryDevice);   % 0/1 uint8

timestamp  = datestr(now, 'yyyymmdd_HHMMSS');
fname      = sprintf('DMD_mask_%s_n%d_%s_r%.3f_%s.mat', ...
                 target_mode, n, ...
                 strrep(sprintf('%.4f', Fidelity), '.', 'p'), ...
                 r_norm, timestamp);
script_dir = fileparts(mfilename('fullpath'));
if isempty(script_dir), script_dir = pwd; end
fpath      = fullfile(script_dir, fname);

save(fpath, 'dmd_mask', 'n', 'Nsp', 'Ndmd', ...
     'target_mode', 'lambda', 'd', 'f_L2', ...
     'f0x_norm', 'f0y_norm', 'r_norm', 'Fidelity', ...
     'x_ap', 'y_ap', 'r_phys');

fprintf('\nMasque sauvegardé : %s\n', fpath)
fprintf('  Taille  : %d × %d  (uint8, 0/1)\n', Ndmd, Ndmd)
fprintf('  Fidélité: %.4f\n', Fidelity)

function s = ternary_str(cond, a, b)
    if cond, s=a; else, s=b; end
end
