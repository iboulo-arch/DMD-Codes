clear; close all; clc;

%% ========================================================
%  BINARY PHASE MODULATION VIA THE LEE HOLOGRAM METHOD
%  (no macro-pixels: phase is encoded pixel-by-pixel)
%
%  References:
%   - W.H. Lee, Progress in Optics (1978)
%   - D.B. Conkey et al., Opt. Express (2012)
%   - https://www.wavefrontshaping.net/post/id/16
%
%  PRINCIPLE
%   A target phase phi(x,y), sampled at every DMD pixel, is
%   encoded into a binary AMPLITUDE grating:
%       f(x,y) = 1/2 * [1 + cos(2*pi*nu0*(x-y) - phi(x,y))]
%       g(x,y) = 1 if f(x,y) > 1/2, else 0          (-> DMD pattern)
%
%   f(x,y) is the sum of 3 terms centered at 3 distinct spatial
%   frequencies: 0 order (0,0), +1 order (nu0,-nu0), -1 order
%   (-nu0,nu0). If nu0 is larger than the spatial bandwidth of
%   phi(x,y), the +1 order alone reconstructs exp(i*phi(x,y))
%   after pinhole filtering in the Fourier plane.
%
%  RESOLUTION / FIDELITY TRADE-OFF
%   Unlike the superpixel method, there is no spatial block
%   averaging here. The trade-off instead comes from the
%   bandwidth constraint: phi(x,y) must be LOW-PASS FILTERED so
%   its spatial frequency content stays below nu0 (minus the
%   pinhole radius), otherwise the 0/+1/-1 orders overlap and
%   corrupt the reconstruction. A lower cutoff = smoother
%   (lower spatial resolution) target phase but cleaner order
%   separation.
%% ========================================================

%% ========================================================
% PARAMETERS
%% ========================================================

N = 400;                  % DMD grid size (pixels), full resolution, no macro-pixels

nu0 = 0.08;                % carrier spatial frequency (cycles/pixel)
                           % period of the Lee grating = 1/nu0 pixels

phi_cutoff = 0.05;         % max spatial frequency allowed in phi(x,y)
                           % (cycles/pixel) -- must be < nu0 - pinhole_radius

pinhole_radius = 0.035;    % Fourier-plane pinhole radius (cycles/pixel)
                           % isolates the +1 order; must satisfy
                           % phi_cutoff + pinhole_radius < nu0

%% ========================================================
% GENERATE KOLMOGOROV PHASE SCREEN (full resolution, N x N)
%% ========================================================

fprintf('Generating Kolmogorov phase screen...\n')

fx = (-N/2:N/2-1)/N;
[FX,FY] = meshgrid(fx,fx);
f = sqrt(FX.^2 + FY.^2);
f(f==0) = 1e-6;
PSD = f.^(-11/3);

cn = randn(N) + 1i*randn(N);
phi_raw = real(ifft2(ifftshift(cn .* sqrt(PSD))));

%% ========================================================
% LOW-PASS FILTER phi TO RESPECT THE LEE BANDWIDTH CONDITION
%   -> without this, phi's own high spatial frequencies would
%      overlap the +1/-1 orders and corrupt the reconstruction
%% ========================================================

lowpass_mask = (f < phi_cutoff);

Phi_spec = fftshift(fft2(phi_raw)) .* lowpass_mask;
phi = real(ifft2(ifftshift(Phi_spec)));

phi = phi/std(phi(:));
phi = phi*pi;              % target phase, defined at every pixel

%% ========================================================
% BUILD THE LEE HOLOGRAM (pixel by pixel, no macro-pixels)
%% ========================================================

fprintf('Building Lee hologram...\n')

[X,Y] = meshgrid(0:N-1, 0:N-1);

carrier = 2*pi*nu0*(X - Y);

f_lee = 1*(1 + cos(carrier - phi));

%% ========================================================
% BINARIZE -> this is the pattern sent to the DMD
%% ========================================================

DMD = double(f_lee > 0.5);

%% ========================================================
% FOURIER SPECTRUM AND PINHOLE POSITION
%% ========================================================

G = fftshift(fft2(DMD));

% +1 order is centered at (nu_x,nu_y) = (nu0,-nu0), see Eq.(2) in the reference
cx =  nu0;
cy = -nu0;

mask = ((FX-cx).^2 + (FY-cy).^2) < pinhole_radius^2;

%% ========================================================
% APPLY THE PINHOLE FILTER (simulates the 4-f system)
%% ========================================================

fprintf('Filtering +1 diffraction order...\n')

Gfilt = G .* mask;

% recentre the isolated order on (0,0) before inverse FFT (demodulation)
shift_row = round(cy*N);
shift_col = round(cx*N);
Gfilt_centered = circshift(Gfilt, [-shift_row, -shift_col]);

Efield = ifft2(ifftshift(Gfilt_centered));

phi_rec = angle(Efield);

%% ========================================================
% MODULATION FIDELITY (full resolution, pixel by pixel)
%% ========================================================

Et = exp(1i*phi);
Er = Efield;

Et = Et(:)/norm(Et(:));
Er = Er(:)/norm(Er(:));

Fidelity = abs(Et'*Er)^2;

fprintf('\n')
fprintf('=====================================\n')
fprintf('nu0 = %.4f, phi_cutoff = %.4f, pinhole_radius = %.4f\n', ...
    nu0, phi_cutoff, pinhole_radius)
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
title('Kolmogorov target phase (band-limited)')

subplot(2,3,2)
imagesc(DMD)
axis image
colormap(gca,'gray')
title(sprintf('%dx%d Lee binary hologram (\\nu_0=%.3f)', N,N,nu0))

subplot(2,3,3)
imagesc(fx,fx,log(abs(G)+1))
axis image
hold on
theta = linspace(0,2*pi,100);
plot(cx+pinhole_radius*cos(theta), cy+pinhole_radius*sin(theta), 'r-', 'LineWidth',1.5)
plot(0,0,'y+','MarkerSize',10,'LineWidth',1.5)
plot(-cx-pinhole_radius*cos(theta), -cy-pinhole_radius*sin(theta), 'c-', 'LineWidth',1)
colorbar
title('Fourier spectrum + pinhole (red = +1 order kept)')
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
imagesc(abs(Efield))
axis image
colorbar
title('Reconstructed amplitude (should be ~uniform)')

sgtitle(sprintf('Lee Hologram Reconstruction (pixel-wise) — \\nu_0=%.3f — Fidelity=%.4f', ...
    nu0, Fidelity))

%% ========================================================
% PSD COMPARISON
%% ========================================================

figure

PSD_target = abs(fftshift(fft2(phi))).^2;
PSD_rec    = abs(fftshift(fft2(phi_rec))).^2;

loglog(mean(PSD_target(:,ceil(N/2):end),1),'LineWidth',2)
hold on
loglog(mean(PSD_rec(:,ceil(N/2):end),1),'LineWidth',2)
grid on

xlabel('Spatial frequency')
ylabel('PSD')
legend('Target','Reconstructed')
title('Kolmogorov PSD comparison (Lee hologram method)')