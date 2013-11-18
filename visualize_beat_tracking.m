function [] = visualize_beat_tracking( data_fln, movie_fln )
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
%               is of size [n_particles, 5, n_frames]
%
% OUTPUT
% video
% -----------------------------------------------------------------------
addpath('/home/florian/diss/src/matlab/libs/matlab_utils/export_fig')
close all
% thresh =-20;
% -----------------------------------------------------------------------
% parse input arguments
% -----------------------------------------------------------------------
synth_data_ok = 0; hmm_data_ok = 0; pf_data_ok = 0; anns_data_ok = 0;
[path, fname, ~] = fileparts(data_fln);
if isempty(path), path = '~/diss/src/matlab/beat_tracking/bayes_beat/temp'; end
fln = fullfile(path, [fname, '_synth.mat']);
if exist(fln, 'file')
    load(fln);
    synth_data_ok = 1;
    R = 1; % only 1D data
    frame_length = 0.02;
    n_m_points = 70;
    n_n_points = 20;
    M = data.M; N = data.N;
    m_points = round(linspace(1,M,n_m_points));
    n_points = round(linspace(1, N, n_n_points));
    m_grid = repmat(m_points, length(n_points), 1)';
    n_grid = repmat(n_points, length(m_points), 1);
    grid = [m_grid(:), n_grid(:)];
    nFrames = size(data.weights, 2);
end
fln = fullfile(path, [fname, '_hmm.mat']);
if exist(fln, 'file')
    load(fln);
    hmm_data_ok = 1;
    nFrames = size(logP_data, 2);
    [~, nPos, ~] = size(obs_lik);
    max_lik = max(obs_lik(:));
end
fln = fullfile(path, [fname, '_pf.mat']);
if exist(fln, 'file')
    load(fln);
    pf_data_ok = 1;
end
fln = fullfile(path, [fname, '_anns.mat']);
if exist(fln, 'file')
    load(fln);
    anns_data_ok = 1;
end
if synth_data_ok+hmm_data_ok+pf_data_ok == 0
    error('No data found\n')
end

% % -----------------------------------------------------------------------
% % load hmm data
% % -----------------------------------------------------------------------
% load(fullfile('~/diss/src/matlab/beat_tracking/bayes_beat/temp', [fname, '_hmm.mat'])); % load 'logP_data', 'M', 'N', 'R', 'frame_length', 'obs_lik'
% 
% 
% [~, nFrames] = size(logP_data);
% % load pf data
% load(fullfile('~/diss/src/matlab/beat_tracking/bayes_beat/temp', [fname, '_pf.mat'])); % 'logP_data_pf' : [nParticles, R, 3, nFrames]
% % color map
% if length(size(logP_data_pf)) == 4
%     rbpf = 1;
%     temp = logP_data_pf(:, :, 3, :);
%     max_w = max(temp(:));
%     min_w = max([thresh, min(temp(:))]);
%     col_bins = linspace(min_w, max_w, 64);
% elseif length(size(logP_data_pf)) == 3
%     rbpf = 0;
%     temp = logP_data_pf(:, 4, :);
% %     max_w = max(temp(:));
%     max_w = 5;
%     min_w = max([thresh, min(temp(:))]);
%     col_bins = linspace(min_w, max_w, 64);
% end
% load anns data

% -----------------------------------------------------------------------
% initialize graphical display
% -----------------------------------------------------------------------
hf = figure;
% get rid of an annoying warning: "RGB color data not yet supported in Painter's mode."
set(gcf, 'renderer','zbuffer'); 
col = lines;
colormap(gray);
set(gca,'YDir','normal')
nPlots = R * 2;
h_sp = zeros(nPlots, 1);
if exist('movie_fln', 'var')
    movie_fln = fullfile('~/diss/src/matlab/beat_tracking/bayes_beat/temp', movie_fln);
else
    movie_fln = fullfile('~/diss/src/matlab/beat_tracking/bayes_beat/temp', [fname, '.avi']);
end
aviobj = avifile(movie_fln, 'fps', 1/(2*frame_length));
% aviobj = VideoWriter(movie_fln, 'Motion JPEG AVI');
% aviobj.FrameRate = 1/(2*frame_length);
% open(aviobj);
% % ax=get(hf,'Position');

set(hf, 'Position', [51 1 1316 689]);
y_pos = [0.63; 0.18];
% set(hf, 'Position', [51 1 650 345]);
% y_pos = [0.63; 0.18];

% -----------------------------------------------------------------------
% visualize
% -----------------------------------------------------------------------

for iFrame = 1:2:nFrames
% for iFrame = [1, 10, 100, 1000]
    
    %     max_h = max(logP_data(important_pix, iFrame));
    %     min_h = max([thresh, min(logP_data(important_pix, iFrame))]);
    for iR=1:R
        if hmm_data_ok
            important_pix = find(logP_data(:, iFrame));
            plot_id = (iR-1)*R+1;
            h_sp(plot_id) = subplot(nPlots, 1, plot_id);
            start_ind = sub2ind([M, N, R], 1, 1, iR);
            end_ind = sub2ind([M, N, R], M, N, iR);
            frame = reshape(logP_data(start_ind:end_ind, iFrame), M, N);
            important_pix = find(frame);
            max_h = max(frame(important_pix));
            min_h = max([thresh, min(frame(important_pix))]);
            frame(frame(important_pix)<min_h) = min_h;
            frame(important_pix) = frame(important_pix) - thresh;
            imagesc(frame');
            caxis([0 max([1, max_h - thresh - 5])])
        end
        if synth_data_ok
            p = evaluate_p(grid, iFrame);
            p = p/sum(p);
            p_map = reshape(p, n_m_points, n_n_points)';
            imagesc(m_points, n_points, p_map);
            caxis([0 0.025])
            %             set(gca,'YDir','normal')
        end
        colorbar;
        hold on;
        if pf_data_ok
            ind = (logP_data_pf(:, 3, iFrame) == iR);
            if sum(ind) > 0
                for iCluster=unique(logP_data_pf(:, 5, iFrame))'
                    is_in_cluster = (logP_data_pf(:, 5, iFrame) == iCluster);
%                     col_bins = linspace(min(logP_data_pf(ind & is_in_cluster, 4, iFrame)), max(logP_data_pf(ind & is_in_cluster, 4, iFrame)), 10);
%                     [~, bins] = histc(logP_data_pf(ind & is_in_cluster, 4, iFrame), col_bins);
%                     bins(bins < 1) = 1; bins(bins > 10) = 10;
                    scatter(logP_data_pf(ind & is_in_cluster, 1, iFrame), logP_data_pf(ind & is_in_cluster, 2, iFrame), ...
                        10, col(iCluster, :), 'filled');
                end
            end
        end
        if anns_data_ok
            if anns(1,3) == iR
                scatter(anns(iFrame, 1), anns(iFrame, 2), 30, 'c', 'filled');
            end
        end
%         set(gcf, 'renderer','zbuffer'); 
        title(sprintf('Frame: %i', iFrame));
        %         ax=get(h_sp(iR),'Position');
        set(gca,'YDir','normal')
        if ~synth_data_ok
            h_sp(plot_id+1) = subplot(nPlots, 1, plot_id+1);
            stairs(obs_lik(iR, :, iFrame));
            xlim([1 nPos])
            ylim([0 max_lik])
            %         ax=get(h_sp(plot_id+1),'Position');
            set(h_sp(plot_id), 'Position', [0.1 y_pos(iR) 0.8 0.3]); % [xmin ymin xlenght ylength]);
            set(h_sp(plot_id+1), 'Position', [0.1 y_pos(iR)-0.11 0.8 0.08]); % [xmin ymin xlenght ylength]);
        end
        F = getframe(hf);
        aviobj = addframe(aviobj,F);
%         writeVideo(aviobj, F);
    end 
    fln = ['./temp/02_kmeans_', num2str(iFrame), '.pdf'];
    export_fig(fln);
end
close(hf);
aviobj = close(aviobj);
end
