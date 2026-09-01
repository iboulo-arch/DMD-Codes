clear; close all; clc;

%% ========================================================
%  PHYSICAL SUPERPIXEL DMD TURBULENCE SIMULATOR
%
%  This version builds a PHYSICAL LUT including:
%   - Fourier propagation
%   - pinhole filter radius r
%   - coherent overlap
%
%  Therefore:
%   changing r changes the LUT itself
%
%% ========================================================

%% ========================================================
% PARAMETERS
%% ========================================================

Nsp = 100;      % number of superpixels in a row
n   = 4;        % superpixel size

Ndmd = Nsp*n;   % number of pixels in a row

r0 = 15;

%% ========================================================
% FOURIER PINHOLE RADIUS
%% ========================================================

r = 0.064;

%% ========================================================
% SUPERPIXEL PHASES
%% ========================================================

phases = reshape(2*pi*(0:n^2-1)/n^2,n,n);

%% ========================================================
img = imread('barsik_768.jpg');

if size(img,3)==3
    img = rgb2gray(img);
end

img = imresize(img,[Nsp Nsp]);

img = double(img)/255;

phi = pi*(2*img-1);


%% ========================================================

Etarget = exp(1i*phi);

%% ========================================================
% BUILD PHYSICAL LUT
%% ========================================================

fprintf('Building PHYSICAL LUT...\n')

Nstates = 2^(n^2);

Estate = zeros(Nstates,1);

Bits = false(Nstates,n^2);

%% --------------------------------------------------------
% Fourier coordinates for one superpixel
%% --------------------------------------------------------

pad = 64;

fx = (-pad/2:pad/2-1)/pad;

[FX,FY] = meshgrid(fx,fx);

FR = sqrt(FX.^2 + FY.^2);

%% --------------------------------------------------------
% Physical pinhole filter
%% --------------------------------------------------------

H = double(FR < r);

%% ========================================================
% BUILD LUT
%% ========================================================

for s = 0:Nstates-1

    %% ----------------------------------------------------
    % binary pattern
    %% ----------------------------------------------------

    bits = logical(bitget(s,1:n^2));

    Bits(s+1,:) = bits;

    block = reshape(bits,n,n);

    %% ----------------------------------------------------
    % complex-valued superpixel
    %% ----------------------------------------------------

    complex_block = block .* exp(1i*phases);

    %% ----------------------------------------------------
    % zero padding
    %% ----------------------------------------------------

    tmp = zeros(pad);

    c0 = pad/2 - n/2 + 1;

    tmp(c0:c0+n-1,c0:c0+n-1) = complex_block;

    %% ----------------------------------------------------
    % Fourier propagation
    %% ----------------------------------------------------

    F = fftshift(fft2(tmp));

    %% ----------------------------------------------------
    % pinhole filter
    %% ----------------------------------------------------

    Ffilt = F .* H;

    %% ----------------------------------------------------
    % reconstructed field
    %% ----------------------------------------------------

    recon = ifft2(ifftshift(Ffilt));

    %% ----------------------------------------------------
    % measure central complex field
    %% ----------------------------------------------------

    Estate(s+1) = recon(pad/2,pad/2);

end

%% --------------------------------------------------------
% normalize LUT
%% --------------------------------------------------------

Estate = Estate / max(abs(Estate));

%% ========================================================
% DISPLAY LUT
%% ========================================================

figure

plot(real(Estate),imag(Estate),'.')

axis equal
grid on

xlabel('Re(E)')
ylabel('Im(E)')

title(sprintf('Physical LUT for r = %.3f',r))

%% ========================================================
% DMD RECONSTRUCTION
%% ========================================================

fprintf('Reconstructing DMD...\n')

DMD = zeros(Ndmd,Ndmd);

Ereconstructed = zeros(Nsp,Nsp);

parfor iy = 1:Nsp

    Erow = zeros(1,Nsp);

    DMDrow = cell(1,Nsp);

    for ix = 1:Nsp

        target = Etarget(iy,ix);

        %% -----------------------------------------------
        % nearest physical LUT state
        %% -----------------------------------------------

        [~,idx] = min(abs(Estate-target));

        Erow(ix) = Estate(idx);

        bits = Bits(idx,:);

        block = reshape(bits,n,n);

        DMDrow{ix} = block;

    end

    Ereconstructed(iy,:) = Erow;

    Rows{iy} = DMDrow;

end

%% ========================================================
% ASSEMBLE FULL DMD
%% ========================================================

for iy = 1:Nsp
    for ix = 1:Nsp

        ys = (iy-1)*n + 1;
        xs = (ix-1)*n + 1;

        DMD(ys:ys+n-1,xs:xs+n-1) = Rows{iy}{ix};

    end
end

%% ========================================================
% FULL OPTICAL PROPAGATION
%% ========================================================

fprintf('Applying full optical propagation...\n')

%% --------------------------------------------------------
% build complex DMD
%% --------------------------------------------------------

ComplexDMD = zeros(Ndmd,Ndmd);

for iy = 1:Nsp
    for ix = 1:Nsp

        ys = (iy-1)*n + 1;
        xs = (ix-1)*n + 1;

        block = DMD(ys:ys+n-1,xs:xs+n-1);

        ComplexDMD(ys:ys+n-1,xs:xs+n-1) = ...
            block .* exp(1i*phases);

    end
end

%% --------------------------------------------------------
% Fourier coordinates
%% --------------------------------------------------------

fx = (-Ndmd/2:Ndmd/2-1)/Ndmd;

[FX,FY] = meshgrid(fx,fx);

FR = sqrt(FX.^2 + FY.^2);

%% --------------------------------------------------------
% physical pinhole
%% --------------------------------------------------------

Hfull = double(FR < r);

%% --------------------------------------------------------
% propagation
%% --------------------------------------------------------

F = fftshift(fft2(ComplexDMD));

Ffilt = F .* Hfull;

ReconFull = ifft2(ifftshift(Ffilt));

%% ========================================================
% EXTRACT SUPERPIXELS
%% ========================================================

Efinal = zeros(Nsp,Nsp);

for iy = 1:Nsp
    for ix = 1:Nsp

        ys = (iy-1)*n + 1;
        xs = (ix-1)*n + 1;

        block = ReconFull(ys:ys+n-1,xs:xs+n-1);

        Efinal(iy,ix) = mean(block(:));

    end
end

%% ========================================================
% PHASES
%% ========================================================

phi_rec = angle(Efinal);

%% ========================================================
% MODULATION FIDELITY
%% ========================================================

Et = Etarget(:);
Er = Efinal(:);

Et = Et / norm(Et);
Er = Er / norm(Er);

Fidelity = abs(Et' * Er)^2;

fprintf('\n')
fprintf('=====================================\n')
fprintf('r = %.4f\n',r)
fprintf('Fidelity = %.5f\n',Fidelity)
fprintf('Error = %.5f\n',1-Fidelity)
fprintf('=====================================\n')

%% ========================================================
% DISPLAY
%% ========================================================

figure('Position',[100 100 1800 900])

subplot(2,3,1)
imagesc(phi)
title('image target phase (continue)')   % phase réelle, peut dépasser [-π, π]
colorbar


subplot(2,3,2)
imagesc(angle(Etarget))
title('Phase target wrapped [-π, π]')        % repliée dans [-π, π] par angle()
colorbar

subplot(2,3,3)
imagesc(abs(Etarget))
axis image
colorbar
title('Target amplitude')

subplot(2,3,4)
imagesc(DMD)
axis image
colormap(gray)
title(sprintf('%dx%d Superpixels ,%dx%d binary DMD',  Nsp , Nsp, Ndmd,Ndmd))

subplot(2,3,5)
imagesc(phi_rec)
axis image
colorbar
title('Reconstructed phase')

subplot(2,3,6)
imagesc(angle(exp(1i*(phi-phi_rec))))
axis image
colorbar
title('Wrapped phase error')

sgtitle(sprintf('Physical Superpixel Reconstruction — r = %.3f — Fidelity = %.4f', ...
    r,Fidelity))

%% ========================================================
% PSD COMPARISON
%% ========================================================

figure

PSD_target = abs(fftshift(fft2(phi))).^2;

PSD_rec = abs(fftshift(fft2(phi_rec))).^2;

loglog(mean(PSD_target(:,Nsp/2:end),1),'LineWidth',2)

hold on

loglog(mean(PSD_rec(:,Nsp/2:end),1),'LineWidth',2)

grid on

xlabel('Spatial frequency')
ylabel('PSD')

legend('Target','Reconstructed')

title('Image PSD comparison')