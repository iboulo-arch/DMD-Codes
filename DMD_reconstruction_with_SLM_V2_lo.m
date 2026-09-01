%% ========================================================
% FULL PHYSICAL DMD / SLM SUPERPIXEL SIMULATOR
%
% This version keeps EXACTLY the same physical propagation
% pipeline as the original DMD script:
%
%   binary device
%      ->
%   full Fourier propagation
%      ->
%   pinhole filtering
%      ->
%   reconstructed complex field
%
% The only difference:
%
%   DMD mode:
%       real superpixel phasors
%
%   SLM mode:
%       virtual tilted mirrors synthesized with
%       local blaze ramps
%
%% ========================================================

clear; close all; clc;

%% ========================================================
% MODES
%% ========================================================

%mode = 'DMD';
mode = 'SLM';

%% ========================================================
% GENERAL PARAMETERS (for the DMD and SLM ?)
%% ========================================================

Nsp  = 100;
n    = 4;
Ndmd = 400;        % INDÉPENDANT de Nsp

nb       = floor(Ndmd / Nsp);
Ndmd_eff = nb * Nsp;

if nb < 1
    error('Ndmd=%d trop petit pour Nsp=%d', Ndmd, Nsp);
end

lambda = 633e-9;

%% ========================================================
% DMD PARAMETERS
%% ========================================================

DMD_pitch = 13.7e-6;

%% ========================================================
% SLM PARAMETERS
%% ========================================================

SLM_pitch = 9.2e-6;

SLM_per_DMD = 4;

theta = 12*pi/180;

%% ========================================================
% FOURIER PINHOLE
%% ========================================================

r = 0.065;

%% ========================================================
% SUPERPIXEL PHASES
%% ========================================================

phases = reshape(2*pi*(0:n^2-1)/n^2,n,n);

%% ========================================================
% GENERATE KOLMOGOROV PHASE SCREEN
%% ========================================================

fprintf('Generating turbulence...\n')

fx = (-Nsp/2:Nsp/2-1)/Nsp;

[FX,FY] = meshgrid(fx,fx);

f = sqrt(FX.^2 + FY.^2);

f(f==0)=1e-6;

PSD = f.^(-11/3);

cn = randn(Nsp)+1i*randn(Nsp);

phi = real(ifft2(ifftshift(cn .* sqrt(PSD))));

phi = phi/std(phi(:));

phi = phi*pi;

%% ========================================================
% TARGET COMPLEX FIELD
%% ========================================================

Etarget = exp(1i*phi);

%% ========================================================
% BUILD PHYSICAL LUT
%% ========================================================

fprintf('Building physical LUT...\n')

Nstates = 2^(n^2);

Estate = zeros(Nstates,1);

Bits = false(Nstates,n^2);

pad = 128;

fx = (-pad/2:pad/2-1)/pad;

[FX,FY] = meshgrid(fx,fx);

FR = sqrt(FX.^2 + FY.^2);

H = double(FR < r);

%% ========================================================
% BUILD LUT
%% ========================================================

for s = 0:Nstates-1

    bits = logical(bitget(s,1:n^2));

    Bits(s+1,:) = bits;

    block = reshape(bits,n,n);

    %% ----------------------------------------------------
    % DMD MODE
    %% ----------------------------------------------------

    if strcmp(mode,'DMD')

        complex_block = block .* exp(1i*phases);

    end

    %% ----------------------------------------------------
    % SLM MODE
    %% ----------------------------------------------------

    if strcmp(mode,'SLM')

        bigN = n*SLM_per_DMD;

        complex_block = zeros(bigN,bigN);

        [xs,ys] = meshgrid(0:SLM_per_DMD-1);

        kx = 2*(2*pi/lambda)*sin(theta)*SLM_pitch;

        %rampON  = mod( kx*xs ,2*pi);
        %rampOFF = mod(-kx*xs ,2*pi);
        kxON  = 2*(2*pi/lambda)*sin(theta)*SLM_pitch;

        kxOFF = 4*kxON;
        rampON  = mod( kxON *xs ,2*pi);
        rampOFF = mod( kxOFF*xs ,2*pi);

        for iy2 = 1:n
            for ix2 = 1:n

                ys0 = (iy2-1)*SLM_per_DMD + 1;
                xs0 = (ix2-1)*SLM_per_DMD + 1;

                if block(iy2,ix2)==1

                    local_phase = rampON;

                else

                    local_phase = rampOFF;

                end

                complex_block(ys0:ys0+SLM_per_DMD-1,...
                              xs0:xs0+SLM_per_DMD-1) = ...
                    exp(1i*local_phase);

            end
        end
    end

    %% ----------------------------------------------------
    % ZERO PADDING
    %% ----------------------------------------------------

    tmp = zeros(pad);

    [ny,nx] = size(complex_block);

    c0x = floor(pad/2 - nx/2)+1;
    c0y = floor(pad/2 - ny/2)+1;

    tmp(c0y:c0y+ny-1,c0x:c0x+nx-1) = complex_block;

    %% ----------------------------------------------------
    % PHYSICAL PROPAGATION
    %% ----------------------------------------------------

    F = fftshift(fft2(tmp));

    Ffilt = F .* H;

    recon = ifft2(ifftshift(Ffilt));

    %% ----------------------------------------------------
    % MEASURE CENTRAL FIELD
    %% ----------------------------------------------------

    Estate(s+1) = recon(pad/2,pad/2);

end

%% --------------------------------------------------------
% NORMALIZE LUT
%% --------------------------------------------------------

Estate = Estate/max(abs(Estate));

%% ========================================================
% DISPLAY LUT
%% ========================================================

figure

plot(real(Estate),imag(Estate),'.')

axis equal
grid on

xlabel('Re(E)')
ylabel('Im(E)')

title(sprintf('%s physical LUT',mode))

%% ========================================================
% BUILD DEVICE
%% ========================================================

fprintf('Building device...\n')

if strcmp(mode,'DMD')

   Device = zeros(Ndmd_eff, Ndmd_eff);

end

if strcmp(mode,'SLM')

    Device = zeros(Ndmd*SLM_per_DMD,...
                   Ndmd*SLM_per_DMD);

end

%% ========================================================
% RECONSTRUCTION
%% ========================================================

Ereconstructed = zeros(Nsp,Nsp);

for iy = 1:Nsp

    for ix = 1:Nsp

        target = Etarget(iy,ix);

        [~,idx] = min(abs(Estate-target));

        Ereconstructed(iy,ix) = Estate(idx);

        bits = Bits(idx,:);

        block = reshape(bits,n,n);

        %% ------------------------------------------------
        % DMD MODE
        %% ------------------------------------------------

        if strcmp(mode,'DMD')
            reps     = ceil(nb / n);
            block_nb = repmat(block, reps, reps);
            block_nb = block_nb(1:nb, 1:nb);
            ys = (iy-1)*nb + 1;  xs = (ix-1)*nb + 1;
            Device(ys:ys+nb-1, xs:xs+nb-1) = block_nb;
   
        end

        %% ------------------------------------------------
        % SLM MODE
        %% ------------------------------------------------

        if strcmp(mode,'SLM')

            [xs2,ys2] = meshgrid(0:SLM_per_DMD-1);

            kx = 2*(2*pi/lambda)*sin(theta)*SLM_pitch;

            rampON  = mod( kx*xs2 ,2*pi);
            rampOFF = mod(-kx*xs2 ,2*pi);

            for iy2 = 1:n
                for ix2 = 1:n

                    ys0 = ((iy-1)*n + iy2 -1)*SLM_per_DMD +1;
                    xs0 = ((ix-1)*n + ix2 -1)*SLM_per_DMD +1;

                    if block(iy2,ix2)==1

                        local_phase = rampON;

                    else

                        local_phase = rampOFF;

                    end

                    Device(ys0:ys0+SLM_per_DMD-1,...
                           xs0:xs0+SLM_per_DMD-1) = ...
                        local_phase;

                end
            end
        end
    end
end

%% ========================================================
% BUILD COMPLEX DEVICE
%% ========================================================

fprintf('Applying full physical propagation...\n')

if strcmp(mode,'DMD')

    ComplexDevice = zeros(size(Device));

    for iy = 1:Nsp
        for ix = 1:Nsp
            ph_nb = repmat(phases, reps, reps);
            ph_nb = ph_nb(1:nb, 1:nb);
            ys = (iy-1)*nb + 1;  xs = (ix-1)*nb + 1;
            block = Device(ys:ys+nb-1, xs:xs+nb-1);
            ComplexDevice(ys:ys+nb-1, xs:xs+nb-1) = block .* exp(1i*ph_nb);
        end
    end
end

if strcmp(mode,'SLM')

    ComplexDevice = exp(1i*Device);

end

%% ========================================================
% FULL FOURIER PROPAGATION
%% ========================================================

Ny = size(ComplexDevice,1);

fx = (-Ny/2:Ny/2-1)/Ny;

[FX,FY] = meshgrid(fx,fx);

FR = sqrt(FX.^2 + FY.^2);

Hfull = double(FR < r);

F = fftshift(fft2(ComplexDevice));

Ffilt = F .* Hfull;

ReconFull = ifft2(ifftshift(Ffilt));

%% ========================================================
% EXTRACT RECONSTRUCTED FIELD
%% ========================================================

Efinal = zeros(Nsp,Nsp);

if strcmp(mode,'DMD')

   pitch_extract = nb;

end

if strcmp(mode,'SLM')

    pitch_extract = n*SLM_per_DMD;

end

for iy = 1:Nsp
    for ix = 1:Nsp

        ys = (iy-1)*pitch_extract +1;
        xs = (ix-1)*pitch_extract +1;

        block = ReconFull(ys:ys+pitch_extract-1,...
                          xs:xs+pitch_extract-1);

        Efinal(iy,ix) = mean(block(:));

    end
end

%% ========================================================
% PHASES
%% ========================================================

phi_rec = angle(Efinal);

%% ========================================================
% FIDELITY
%% ========================================================

Et = Etarget(:);
Er = Efinal(:);

Et = Et/norm(Et);
Er = Er/norm(Er);

Fidelity = abs(Et'*Er)^2;

fprintf('\n')
fprintf('=====================================\n')
fprintf('MODE = %s\n',mode)
fprintf('r = %.4f\n',r)
fprintf('Fidelity = %.5f\n',Fidelity)
fprintf('=====================================\n')

%% ========================================================
% DISPLAY
%% ========================================================

figure('Position',[100 100 1800 900])

subplot(2,3,1)
imagesc(phi)
axis image
colorbar
title('Target turbulence phase')

subplot(2,3,2)
imagesc(angle(Etarget))
axis image
colorbar
title('Target phase')

subplot(2,3,3)
imagesc(abs(Etarget))
axis image
colorbar
title('Target amplitude')

subplot(2,3,4)

if strcmp(mode,'DMD')

    imagesc(Device)
    colormap(gray)

else

    imagesc(Device)
    colormap(hsv)

end

axis image
colorbar

title(sprintf('%s device',mode))

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

sgtitle(sprintf('%s Physical Superpixel Reconstruction — Fidelity = %.4f',...
    mode,Fidelity))