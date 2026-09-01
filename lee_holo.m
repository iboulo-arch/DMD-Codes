clear; close all; clc;

%% ========================================================
%  BINARY PHASE MODULATION VIA THE LEE HOLOGRAM METHOD
%
%  References:
%   - W.H. Lee, Progress in Optics (1978)
%   - D.B. Conkey et al., Opt. Express (2012)
%   - https://www.wavefrontshaping.net/post/id/16
%
%  PRINCIPLE
%   A target phase phi(x,y) is encoded into a binary AMPLITUDE
%   grating:
%       f(x,y) = 1/2 * [1 + cos(2*pi*nu0*(x-y) - phi(x,y))]
%       g(x,y) = 1 if f(x,y) > 1/2, else 0            (binarized -> this is DMD)
%
%   The three terms of f(x,y) sit at three distinct spatial
%   frequencies (0 order, +1 order, -1 order). If nu0 is higher
%   than the bandwidth of phi, a pinhole in the Fourier plane
%   isolates the +1 order, which reconstructs exp(i*phi(x,y)).
%
%  TRADE-OFF (as in the reference): nu0 (carrier frequency),
%  the number of DMD pixels per macro-pixel (n), and the pinhole
%  radius must be tuned together: more pixels per macro-pixel =
%  better phase fidelity but lower spatial resolution.
%% ========================================================

%% ========================================================
% PARAMETERS
%% ========================================================

Nsp = 100;       % number of macro-pixels per row (resolution of the phase screen)
n   = 3;        % DMD pixels per macro-pixel (Lee grating period, in pixels)
                % -> with a period of n pixels, ~n distinct phase levels are achievable

Ndmd = Nsp*n;   % full binary DMD pattern size (pixels)

nu0 = 1/n;      % carrier spatial frequency (cycles/pixel), must exceed
           % the bandwidth of phi on the fine grid (~1/(2n) here)

pinhole_radius = 1/(4*n);   % Fourier-plane pinhole radius (cycles/pixel)
                            % must be small enough to isolate the +1 order
                            % from the 0 and -1 orders without cropping it

%% ========================================================
% GENERATE KOLMOGOROV PHASE SCREEN (target, coarse grid)
%% ========================================================

fprintf('Generating Kolmogorov phase screen...\n')

fx = (-Nsp/2:Nsp/2-1)/Nsp;
[FX,FY] = meshgrid(fx,fx);
f = sqrt(FX.^2 + FY.^2);
f(f==0) = 1e-6;
PSD = f.^(-11/3);

cn = randn(Nsp) + 1i*randn(Nsp);
phi = real(ifft2(ifftshift(cn .* sqrt(PSD))));
phi = phi/std(phi(:));
phi = phi*pi;              % target phase, one value per macro-pixel

%% ========================================================
% UPSAMPLE TARGET PHASE TO THE FINE (DMD) GRID
%   -> phase is held constant within each macro-pixel so it does
%      not add spectral content that would leak into the carrier
%% ========================================================

phi_fine = kron(phi, ones(n));   % Ndmd x Ndmd, piecewise-constant

%% ========================================================
% BUILD THE LEE HOLOGRAM
%% ========================================================

fprintf('Building Lee hologram...\n')

[X,Y] = meshgrid(0:Ndmd-1, 0:Ndmd-1);

carrier = 2*pi*nu0*(X - Y);

f_lee = 0.5*(1 + cos(carrier - phi_fine));

%% ========================================================
% BINARIZE -> this is the pattern sent to the DMD
%% ========================================================

DMD = double(f_lee > 0.5);

%% ========================================================
% DISPLAY THE FOURIER SPECTRUM AND PINHOLE POSITION
%% ========================================================

G = fftshift(fft2(DMD));

fxg = (-Ndmd/2:Ndmd/2-1)/Ndmd;
[FXg,FYg] = meshgrid(fxg,fxg);

% +1 order is centered at (nu_x,nu_y) = (-nu0,+nu0), see Eq.(2) in the reference
cx =  -nu0;
cy = nu0;

mask = ((FXg-cx).^2 + (FYg-cy).^2) < pinhole_radius^2;

%% ========================================================
% APPLY THE PINHOLE FILTER (simulates the 4-f system)
%% ========================================================

fprintf('Filtering +1 diffraction order...\n')

Gfilt = G .* mask;

% recentre the isolated order on (0,0) before inverse FFT
shift_row = round(cy*Ndmd);
shift_col = round(cx*Ndmd);
Gfilt_centered = circshift(Gfilt, [-shift_row, -shift_col]);

Efield = ifft2(ifftshift(Gfilt_centered));

%% ========================================================
% EXTRACT ONE COMPLEX VALUE PER MACRO-PIXEL
%% ========================================================

Efinal = zeros(Nsp,Nsp);

for iy = 1:Nsp
    for ix = 1:Nsp
        ys = (iy-1)*n + 1;
        xs = (ix-1)*n + 1;
        block = Efield(ys:ys+n-1, xs:xs+n-1);
        Efinal(iy,ix) = mean(block(:));
    end
end

phi_rec = angle(Efinal);

%% ========================================================
% MODULATION FIDELITY
%% ========================================================

Et = exp(1i*phi);
Er = Efinal;

Et = Et(:)/norm(Et(:));
Er = Er(:)/norm(Er(:));

Fidelity = abs(Et'*Er)^2;

fprintf('\n')
fprintf('=====================================\n')
fprintf('nu0 = %.4f, n = %d, pinhole_radius = %.4f\n', nu0, n, pinhole_radius)
fprintf('Fidelity = %.5f\n', Fidelity)
fprintf('Error = %.5f\n', 1-Fidelity)
fprintf('=====================================\n')

%% ========================================================
% DISPLAY
%% ========================================================

figure('Position',[100 100 1800 900])

subplot(2,3,1)
imagesc(phi)
axis image
colorbar
title('Kolmogorov target phase (coarse grid)')

subplot(2,3,2)
imagesc(DMD)
axis image
colormap(gca,'gray')
title(sprintf('%dx%d Lee binary hologram (%dx%d macro-pixels, n=%d)', ...
    Ndmd,Ndmd,Nsp,Nsp,n))

subplot(2,3,3)
imagesc(fxg,fxg,log(abs(G)+1))
axis image
hold on
theta = linspace(0,2*pi,100);
plot(cx+pinhole_radius*cos(theta), cy+pinhole_radius*sin(theta), 'r-', 'LineWidth',1.5)
colorbar
title('Fourier spectrum of DMD pattern + pinhole (+1 order)')
xlabel('\nu_x (cycles/pixel)')
ylabel('\nu_y (cycles/pixel)')

subplot(2,3,4)
imagesc(phi_rec)
axis image
colorbar
title('Reconstructed phase (after +1 order filtering)')

subplot(2,3,5)
imagesc(angle(exp(1i*(phi-phi_rec))))
axis image
colorbar
title('Wrapped phase error')

subplot(2,3,6)
imagesc(abs(Efinal))
axis image
colorbar
title('Reconstructed amplitude (should be ~uniform)')

sgtitle(sprintf('Lee Hologram Reconstruction — n=%d, \\nu_0=%.3f — Fidelity=%.4f', ...
    n, nu0, Fidelity))

%% ========================================================
% PSD COMPARISON
%% ========================================================

figure

PSD_target = abs(fftshift(fft2(phi))).^2;
PSD_rec    = abs(fftshift(fft2(phi_rec))).^2;

loglog(mean(PSD_target(:,ceil(Nsp/2):end),1),'LineWidth',2)
hold on
loglog(mean(PSD_rec(:,ceil(Nsp/2):end),1),'LineWidth',2)
grid on

xlabel('Spatial frequency')
ylabel('PSD')
legend('Target','Reconstructed')
title('Kolmogorov PSD comparison (Lee hologram method)')