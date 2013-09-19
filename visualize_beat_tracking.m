function [] = visualize_beat_tracking( fname )
%[] = visualize_beat_tracking( data, out_fln )
close all
thresh =-15;
% load hmm data
load(fullfile('~/diss/src/matlab/beat_tracking/bayes_beat/temp', [fname, '_hmm.mat'])); % load 'logP_data', 'M', 'N', 'R', 'frame_length', 'obs_lik'

[~, nPos, ~] = size(obs_lik);
max_lik = max(obs_lik(:));
[~, nFrames] = size(logP_data);
% load pf data
load(fullfile('~/diss/src/matlab/beat_tracking/bayes_beat/temp', [fname, '_pf.mat'])); % 'logP_data_pf' : [nParticles, R, 3, nFrames]
% color map
if length(size(logP_data_pf)) == 4
    rbpf = 1;
    temp = logP_data_pf(:, :, 3, :);
    max_w = max(temp(:));
    min_w = max([thresh, min(temp(:))]);
    col_bins = linspace(min_w, max_w, 64);
elseif length(size(logP_data_pf)) == 3
    rbpf = 0;
    temp = logP_data_pf(:, 4, :);
%     max_w = max(temp(:));
    max_w = 5;
    min_w = max([thresh, min(temp(:))]);
    col_bins = linspace(min_w, max_w, 64);
end
% load anns data
load(fullfile('~/diss/src/matlab/beat_tracking/bayes_beat/temp', [fname, '_anns.mat']));

% initialize video
hf = figure;
col = autumn;
colormap(gray);
set(gca,'YDir','normal')
nPlots = R * 2;
h_sp = zeros(nPlots, 1);
aviobj = VideoWriter(fullfile('~/diss/src/matlab/beat_tracking/bayes_beat/temp', [fname, '.avi']), 'Motion JPEG AVI');
aviobj.FrameRate = 1/(2*frame_length);
% ax=get(hf,'Position');
set(hf, 'Position', [51 1 1316 689]);
open(aviobj);
y_pos = [0.63; 0.18];
for iFrame = 1:2:500
    important_pix = find(logP_data(:, iFrame));
    max_h = max(logP_data(important_pix, iFrame));
    min_h = max([thresh, min(logP_data(important_pix, iFrame))]);
    for iR=1:R
        plot_id = (iR-1)*R+1;
        h_sp(plot_id) = subplot(nPlots, 1, plot_id);
        start_ind = sub2ind([M, N, R], 1, 1, iR);
        end_ind = sub2ind([M, N, R], M, N, iR);
        frame = reshape(logP_data(start_ind:end_ind, iFrame), M, N);
%         frame(frame==0) = min_h;
        important_pix = find(frame);
%         max_h = max(frame(important_pix));
%         min_h = max([thresh, min(frame(important_pix))]);
        frame(frame(important_pix)<min_h) = min_h;
        frame(important_pix) = frame(important_pix) - thresh;
%         frame(find(frame)) = (frame(find(frame)) + min_h) / max_h;
        imagesc(frame');
        caxis([0 max([3, max_h - thresh - 9])])
        colorbar;
        hold on;
        if rbpf
            [~, bins] = histc(logP_data_pf(:, iR, 3, iFrame), col_bins);
            bins(bins < 1) = 1; bins(bins > 64) = 64;
            scatter(logP_data_pf(:, iR, 1, iFrame), logP_data_pf(:, iR, 2, iFrame), ...
                10, col(bins, :), 'filled')
        else
            ind = (logP_data_pf(:, 3, iFrame) == iR);
            if sum(ind) > 0
            col_bins = linspace(min(logP_data_pf(ind, 4, iFrame)), max(logP_data_pf(ind, 4, iFrame)), 64);
            [~, bins] = histc(logP_data_pf(ind, 4, iFrame), col_bins);
            bins(bins < 1) = 1; bins(bins > 64) = 64;
            scatter(logP_data_pf(ind, 1, iFrame), logP_data_pf(ind, 2, iFrame), ...
                10, col(bins, :), 'filled');
            end
%             caxis(h, [min(logP_data_pf(ind, 1, iFrame)), max(logP_data_pf(ind, 1, iFrame))])
        end
        if anns(1,3) == iR
            scatter(anns(iFrame, 1), anns(iFrame, 2), 30, 'c', 'filled');
        end
        title(sprintf('Frame: %i', iFrame));
        %         ax=get(h_sp(iR),'Position');
        %
        set(gca,'YDir','normal')
        
        h_sp(plot_id+1) = subplot(nPlots, 1, plot_id+1);
        stairs(obs_lik(iR, :, iFrame));
        xlim([1 nPos])
        ylim([0 max_lik])
        %         ax=get(h_sp(plot_id+1),'Position');
        set(h_sp(plot_id), 'Position', [0.1 y_pos(iR) 0.8 0.3]); % [xmin ymin xlenght ylength]);
        set(h_sp(plot_id+1), 'Position', [0.1 y_pos(iR)-0.11 0.8 0.08]); % [xmin ymin xlenght ylength]);
    end
    
    F = getframe(hf);
    writeVideo(aviobj, F);
end
close(aviobj);
close(hf);
end

