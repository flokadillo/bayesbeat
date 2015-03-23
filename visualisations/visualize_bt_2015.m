function [] = visualize_bt_2015( data_fln, anns_file, movie_fln )
% VISUALIZE_BEAT_TRACKING visualize (filtering) posterior probability of the hidden
% states given the observations
%
% INPUT:
%   data_fln:  filename of fixed grid (FG) and/or particle grid (PG) data
%           FG has the ending '_hmm.mat' or '_func.mat'
%               '_hmm.mat' has the fields
%                   'logP_data', 'M', 'N', 'R', 'frame_length', 'obs_lik'
%                   logP_data is a sparse matrix of size [n_states x n_frames]
%               '_func.mat' contains the variables 'data' and 'evaluate_p'
%           PG has the ending '_pf.mat' and contains 'logP_data_pf' which
%               is of size [n_particles, 5, n_frames]:
%               where dim 2 is : m x n x r x w x group
%
% OUTPUT
% video
% -----------------------------------------------------------------------
% -----------------------------------------------------------------------
% parse input arguments
% -----------------------------------------------------------------------

[~, ~, ext] = fileparts(data_fln);
if exist(data_fln, 'file')
    if strcmp(ext, '.mat')
        load(data_fln); % load alpha_mat, m, n, r, best_states
        R = 4;
%         [R, ~, ~] = size(obs_lik);
    elseif strcmp(ext, '.h5')
        alpha_mat = log(h5read(data_fln, '/alpha_mat'));
        best_states = h5read(data_fln, '/best_states');
        M  = h5read(data_fln, '/M');
        frame_length = h5read(data_fln, '/frame_length');
        m = h5read(data_fln, '/m');
        n = h5read(data_fln, '/n');
        r = h5read(data_fln, '/r');
        y = h5read(data_fln, '/y');
        R = 4;
    end
    nFrames = size(alpha_mat, 2); 
else
    fprintf('%s not found\n', fln);
    return
end
if exist(anns_file, 'file')
    load(anns_file);
    if strfind(data_fln, 'r4a')
        anns(:, 3) = 3;
    elseif strfind(data_fln, 'r4b')
        anns(:, 3) = 4;
    end
    nFrames = min([nFrames, size(anns, 1)]);
    anns_data_ok = 1;
else
    anns_data_ok = 0;
end
% -----------------------------------------------------------------------
% initialize graphical display
% -----------------------------------------------------------------------
hf = figure;
% get rid of an annoying warning: "RGB color data not yet supported in Painter's mode."
set(gcf, 'renderer','zbuffer');
set(gca,'YDir','normal')
nPlots = R * 2 + 2;
n_cols = ceil(sqrt(R));
n_rows = ceil(R / n_cols) + 1;
h_sp = zeros(nPlots, 1);
aviobj = VideoWriter(movie_fln);
aviobj.FrameRate = 1/(frame_length);
open(aviobj);
% % ax=get(hf,'Position');

% size/resolution of the figure
set(hf, 'Position', [1 1 1000 500]);
% -----------------------------------------------------------------------
% visualize
% -----------------------------------------------------------------------
if exist('obs_lik', 'var')
    obs = obs_lik(:, :, 1); 
    n_gmms = zeros(R, 1);
    for iR=1:R
        n_gmms(iR) = length(obs(iR, obs(iR, :) >= 0)); 
    end
end
plot_pos = [1, 2, 5, 6];
for iFrame = 1:5:nFrames
    max_h = max(alpha_mat(:, iFrame));
    min_h = max([min(alpha_mat(~isinf(alpha_mat(:, iFrame)), iFrame)), max_h - 50]);
    alpha = alpha_mat(:, iFrame);
    alpha(alpha < min_h) = min_h;
    if exist('obs_lik', 'var')
        obs = obs_lik(:, :, iFrame);
        max_obs = log(max(obs(:))) + 1;
        min_obs = log(max([min(obs(obs > 0)), eps])) - 1;
    end
    
    for iR=1:R
        plot_id = plot_pos(iR);
        % hmm probability
        h_sp(plot_id) =  subplot(2 * n_rows + 1, n_cols, plot_id);
        idx = (r == iR);
        scatter(m(idx), n(idx), 50, alpha(idx), 'filled');
        ylim([min(n(idx)), max(n(idx))])
        xlim([1, max(m(idx))])
        caxis([min_h max_h])
        colorbar;
        hold on;
        % best_state
        if r(best_states(iFrame)) == iR
            scatter(m(best_states(iFrame)), n(best_states(iFrame)), 200, 'w', ...
                'filled', 'MarkerEdgeColor', 'k');
        end
        % groundtruth
        if anns_data_ok && (anns(iFrame, 3) == iR)
            scatter(anns(iFrame, 1), anns(iFrame, 2), 100, 'r', 'filled', ...
                'MarkerEdgeColor', 'k');
        end
        hold off;
        % observation probability
        plot_id = plot_id + n_cols;
        h_sp(plot_id) =  subplot(2 * n_rows + 1, n_cols, plot_id);
        if exist('obs_lik', 'var')
            obs_pos = obs(iR, (obs(iR, :) >= 0));
            stairs(log(obs_pos));
            xlim(h_sp(plot_id), [1 n_gmms(iR)+1])
            ylim(h_sp(plot_id), [min_obs max_obs])
        end
    end
    h_sp(end) = subplot(2 * n_rows + 1, n_cols, [nPlots-1 nPlots]);
    if (iFrame > 50) && (iFrame < nFrames - 50)
        plot(y(iFrame-49:iFrame+50))
        hold on;
        scatter(50, y(iFrame), 50, 'b');
        hold off
    else
        plot(y(1:iFrame))
        hold on
        scatter(iFrame, y(iFrame), 50, 'b');
        hold off
    end
    xlim([1 100])
    xlabel(sprintf('Frame: %i (red=groundtruth, white=estimated)', iFrame))  
    % arrange in figure window
    set(h_sp(1), 'Position', [0.03 0.75 0.4 0.2]);
    set(h_sp(2), 'Position', [0.53 0.75 0.4 0.2]);
    set(h_sp(3), 'Position', [0.03 0.61 0.4 0.1]);
    set(h_sp(4), 'Position', [0.53 0.61 0.4 0.1]);
    set(h_sp(5), 'Position', [0.03 0.36 0.4 0.2]);
    set(h_sp(6), 'Position', [0.53 0.36 0.4 0.2]);
    set(h_sp(7), 'Position', [0.03 0.22 0.4 0.1]);
    set(h_sp(8), 'Position', [0.53 0.22 0.4 0.1]);
    set(h_sp(10), 'Position', [0.03 0.08 0.94 0.1]);
    F = getframe(hf);
    writeVideo(aviobj, F);
    clf;
end
close(hf);
close(aviobj);

% % prepare audio file
% fln = fullfile('~/diss/data/beats/boeck', [fname, '.wav']);
% if exist(fln)
%     [y, fs] = wavread(fln );
%     y = y(1:nFrames*frame_length*fs);
%     wavwrite(y, fs, 'test.wav');
%     system(['lame test.wav ', fullfile(path, [fname, '.mp3'])]);
%     fprintf('Saved cut mp3 to %s\n', fullfile(path, [fname, '.mp3']));
%     [path, fname, ~] = fileparts(movie_fln);
%     %     cmd = ['ffmpeg -i ', movie_fln, ' -b 9600 -qscale 5 -acodec copy -i ', fullfile(path, [fname, '.mp3']), ...
%     %         ' ', fullfile(path, [fname, '_av.avi'])];
%     %     fprintf('Executing %s\n', cmd);
%     %     system(cmd);
% end
end
