clear; close all; clc;

%% ========================================================
%  DMD Superpixel interactif
%
%  Figure 1 :
%   - image des phases
%   - intensité résultante
%
%  Figure 2 :
%   - sélection interactive des pixels ON/OFF
%   - somme complexe type Feynman
%
%% ========================================================

n = 4;

%% ========================================================
% phases des pixels
%% ========================================================

phases = reshape(2*pi*(0:n^2-1)/n^2,n,n);

%% ========================================================
% état ON/OFF
%% ========================================================

ON = zeros(n,n);

%% ========================================================
% paramètres visuels
%% ========================================================

scaleVisu = 1.4;

theta = linspace(0,2*pi,300);

%% ========================================================
% figures
%% ========================================================

fig1 = figure( ...
    'Name','Image et Intensite', ...
    'Position',[50 100 900 450]);

fig2 = figure( ...
    'Name','Selection DMD + Feynman', ...
    'Position',[1000 100 1000 500]);

%% ========================================================
% boucle interactive
%% ========================================================

while ishandle(fig1) && ishandle(fig2)

    %% ====================================================
    % CALCUL CHAMP COMPLEXE
    %% ====================================================

    E = 0;

    for iy = 1:n
        for ix = 1:n

            if ON(iy,ix)==1

                E = E + exp(1i*phases(iy,ix));

            end
        end
    end

    amp = abs(E);
    inten = abs(E)^2;
    phaseE = angle(E);

    %% ====================================================
    % FIGURE 1
    %% ====================================================

    figure(fig1)
    clf

    %% ----------------------------------------------------
    % phases des pixels
    %% ----------------------------------------------------

    subplot(1,2,1)

    imagesc(phases/pi)

    axis image
    colorbar

    title('Phases des pixels (\times\pi)')

    for iy = 1:n
        for ix = 1:n

            txt = sprintf('%.2f',phases(iy,ix)/pi);

            text(ix,iy,txt,...
                'HorizontalAlignment','center',...
                'Color','w',...
                'FontWeight','bold');
        end
    end

    %% ----------------------------------------------------
    % intensité résultante
    %% ----------------------------------------------------

    subplot(1,2,2)

    hold on
    axis equal
    grid on

    title('Champ complexe résultant')

    xlabel('Re(E)')
    ylabel('Im(E)')

    % cercle unité
    plot(cos(theta),sin(theta),'k--')

    % vecteur total
    quiver(0,0,...
        scaleVisu*real(E),...
        scaleVisu*imag(E),...
        0,...
        'r',...
        'LineWidth',5,...
        'MaxHeadSize',1.5);

    % étoile finale
    plot(scaleVisu*real(E),...
         scaleVisu*imag(E),...
         'rp',...
         'MarkerSize',24,...
         'MarkerFaceColor','r')

    % texte
    text(-7,6,sprintf('Amplitude = %.2f',amp),...
        'FontSize',14)

    text(-7,5,sprintf('Intensite = %.2f',inten),...
        'FontSize',14)

    text(-7,4,sprintf('Phase = %.2f rad',phaseE),...
        'FontSize',14)

    xlim([-8 8])
    ylim([-8 8])

    %% ====================================================
    % FIGURE 2
    %% ====================================================

    figure(fig2)
    clf

    %% ----------------------------------------------------
    % sélection ON/OFF
    %% ----------------------------------------------------

    subplot(1,2,1)

    imagesc(ON)

    axis image

    colormap(gray)

    title('Cliquer pour ON/OFF')

    hold on

    for iy = 1:n
        for ix = 1:n

            % afficher phase
            txt = sprintf('%.2f',phases(iy,ix)/pi);

            text(ix,iy,txt,...
                'HorizontalAlignment','center',...
                'Color','w',...
                'FontWeight','bold');

            % contour vert ON
            if ON(iy,ix)==1

                rectangle( ...
                    'Position',[ix-0.5 iy-0.5 1 1], ...
                    'EdgeColor','g', ...
                    'LineWidth',4);

            end
        end
    end

    %% ----------------------------------------------------
    % diagramme Feynman
    %% ----------------------------------------------------

    subplot(1,2,2)

    hold on
    axis equal
    grid on

    title('Addition des phasors')

    xlabel('Re(E)')
    ylabel('Im(E)')

    plot(cos(theta),sin(theta),'k--')

    colors = hsv(n^2);

    origin = 0;

    k = 1;

    for iy = 1:n
        for ix = 1:n

            if ON(iy,ix)==1

                z = exp(1i*phases(iy,ix));

                new_origin = origin + z;

                %% ----------------------------------------
                % flèche cumulative
                %% ----------------------------------------

                quiver( ...
                    scaleVisu*real(origin),...
                    scaleVisu*imag(origin),...
                    scaleVisu*real(z),...
                    scaleVisu*imag(z),...
                    0,...
                    'Color',colors(k,:),...
                    'LineWidth',4,...
                    'MaxHeadSize',1.5);

                %% ----------------------------------------
                % points
                %% ----------------------------------------

                plot(scaleVisu*real(origin),...
                     scaleVisu*imag(origin),...
                     'ko',...
                     'MarkerSize',6,...
                     'MarkerFaceColor','k')

                plot(scaleVisu*real(new_origin),...
                     scaleVisu*imag(new_origin),...
                     '.',...
                     'Color',colors(k,:),...
                     'MarkerSize',30)

                %% ----------------------------------------
                % numéro pixel
                %% ----------------------------------------

                text(scaleVisu*real(new_origin),...
                     scaleVisu*imag(new_origin),...
                     sprintf('%d',k),...
                     'FontSize',10,...
                     'FontWeight','bold')

                origin = new_origin;

                k = k + 1;

            end
        end
    end

    %% ----------------------------------------------------
    % résultat final
    %% ----------------------------------------------------

    plot(scaleVisu*real(E),...
         scaleVisu*imag(E),...
         'rp',...
         'MarkerSize',26,...
         'MarkerFaceColor','r')

    xlim([-8 8])
    ylim([-8 8])

    %% ====================================================
    % interaction souris
    %% ====================================================

    subplot(1,2,1)

    [xclick,yclick,button] = ginput(1);

    if isempty(button)
        break
    end

    ix = round(xclick);
    iy = round(yclick);

    if ix>=1 && ix<=n && iy>=1 && iy<=n

        ON(iy,ix) = 1 - ON(iy,ix);

    end
end