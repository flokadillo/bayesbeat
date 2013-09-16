function [] = visualize_beat_tracking( in_fln, out_fln )
%[] = visualize_beat_tracking( data, out_fln )
close all
% load data
load(in_fln); % load 'logP_data', 'M', 'N', 'R', 'frame_length'
[~, nPos, ~] = size(obs_lik);
max_lik = max(obs_lik(:));
[nStates, nFrames] = size(logP_data);

% initialize video
hf = figure;
set(gca,'YDir','normal')
nPlots = R * 2;
h_sp = zeros(nPlots, 1);
aviobj = VideoWriter(out_fln, 'Uncompressed AVI');
aviobj.FrameRate = 1/frame_length;
% ax=get(hf,'Position');
set(hf, 'Position', [51 1 1316 689]);
open(aviobj);
y_pos = [0.63; 0.18];
for iFrame = 1:300
    for iR=1:R
        
        plot_id = (iR-1)*R+1;
        h_sp(plot_id) = subplot(nPlots, 1, plot_id);
        start_ind = sub2ind([M, N, R], 1, 1, iR);
        end_ind = sub2ind([M, N, R], M, N, iR);
        frame = reshape(logP_data(start_ind:end_ind, iFrame), M, N);
        frame(find(frame)) = (frame(find(frame)) / 10) + 1;
        imagesc(frame', [0 1]);
        colorbar;
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

