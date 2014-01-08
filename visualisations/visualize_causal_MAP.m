function [] = visualize_causal_MAP(fname)

close all

% load detections
load(['./temp/', fname, '_map.mat']);

% load annotations
load(['./temp/', fname, '_anns.mat']);

% load meta data
load(['./temp/', fname, '_hmm.mat']);

movie_fln = ['./temp/', fname, '_causal_map.avi'];

aviobj = avifile(movie_fln, 'fps', 1/frame_length);
% size/resolution of the figure
hf = figure;
set(hf, 'Position', [51 1 500 600]);

display_frame_size = 5; % frame that is displayed in video in seconds
n_frames_display = round(display_frame_size/frame_length);

% until window is filled

suptitle(fname)

for iFrame=1:n_frames_display
    [m, n, r] = backtrace(iFrame, best_state, psi_mat, minState, M, N, R);
    
    subplot(5, 1, 1)
    o = [y(1:iFrame); zeros(n_frames_display-iFrame, 1)];
    plot(o)
    ylim([min(y), max(y)])
    ylabel('Onset feature')
    
    subplot(5, 1, 2)
    m_d = [m; zeros(n_frames_display-iFrame, 1)];
    m_a = [anns(1:iFrame, 1); zeros(n_frames_display-iFrame, 1)];
    plot(m_d)
    hold on
    plot(m_a, 'r')
    hold off
    ylim([0, max(dets(:, 1))])
    ylabel('Bar position')
    
    subplot(5, 1, 3)
    n_d = [n; zeros(n_frames_display-iFrame, 1)];
    n_a = [anns(1:iFrame, 2); zeros(n_frames_display-iFrame, 1)];
    plot(n_d)
    hold on
    plot(n_a, 'r')
    hold off
    xlim([1, n_frames_display])
    ylim([min([dets(:, 2); anns(:, 2)]), max([dets(:, 2); anns(:, 2)])])
    ylabel('Tempo')
    
    subplot(5, 1, 4)
    r_d = [r; zeros(n_frames_display-iFrame, 1)];
    r_a = [anns(1:iFrame, 3); zeros(n_frames_display-iFrame, 1)];
    plot(r_d)
    hold on
    plot(r_a, 'r--')
    hold off
    xlim([1, n_frames_display])
    ylim([1, R])
    ylabel('Rhyth. index')
    legend('detected', 'groundtruth')
    
    subplot(5, 1, 5)
    plot([mean_params(sub2ind([R, size(mean_params, 2)], r, ceil(m/22.5))); zeros(n_frames_display-iFrame, 1)])
    ylim([min(mean_params(:)), max(mean_params(:))])
    ylabel('Rhyth. mean')
    xlabel('Frame')
    
    F = getframe(hf);
    aviobj = addframe(aviobj,F);
    
    
end

for iFrame=2:(size(anns, 1)-n_frames_display)
    [m, n, r] = backtrace(iFrame+n_frames_display-1, best_state, psi_mat, minState, M, N, R);
    
    subplot(5, 1, 1)
    plot(y(iFrame:iFrame+n_frames_display-1))
    ylim([min(y), max(y)])
    %        xlim([iFrame, iFrame+n_frames_display-1])
    set(gca, 'XTick', [0:50:n_frames_display])
    set(gca, 'XTickLabel', strread(num2str([iFrame:50:iFrame+n_frames_display]),'%s'))
    ylabel('Onset feature')
    
    %        b = strread(num2str(a),'%s')
    subplot(5, 1, 2)
    plot(m(iFrame:iFrame+n_frames_display-1))
    hold on
    plot(anns(iFrame:iFrame+n_frames_display-1, 1), 'r')
    hold off
    ylim([0, max(dets(:, 1))])
    set(gca, 'XTick', [0:50:n_frames_display])
    set(gca, 'XTickLabel', strread(num2str([iFrame:50:iFrame+n_frames_display]),'%s'))
    ylabel('Bar position')
    
    subplot(5, 1, 3)
    plot(n(iFrame:iFrame+n_frames_display-1))
    hold on
    plot(anns(iFrame:iFrame+n_frames_display-1, 2), 'r')
    hold off
    ylim([min([dets(:, 2); anns(:, 2)]), max([dets(:, 2); anns(:, 2)])])
    set(gca, 'XTick', [0:50:n_frames_display])
    set(gca, 'XTickLabel', strread(num2str([iFrame:50:iFrame+n_frames_display]),'%s'))
    ylabel('Tempo')
    
    subplot(5, 1, 4)
    plot(r(iFrame:iFrame+n_frames_display-1))
    hold on
    plot(anns(iFrame:iFrame+n_frames_display-1, 3), 'r--')
    hold off
    ylim([1, R])
    set(gca, 'XTick', [0:50:n_frames_display])
    set(gca, 'XTickLabel', strread(num2str([iFrame:50:iFrame+n_frames_display]),'%s'))
    ylabel('Rhyth. index')
    legend('detected', 'groundtruth')
    
    subplot(5, 1, 5)
    plot(mean_params(sub2ind([R, size(mean_params, 2)], ...
        r(iFrame:iFrame+n_frames_display-1), ceil(m(iFrame:iFrame+n_frames_display-1)/22.5))))
%     plot(mean_params(iFrame:iFrame+n_frames_display-1))
    ylim([min(mean_params(:)), max(mean_params(:))])
    set(gca, 'XTick', [0:50:n_frames_display])
    set(gca, 'XTickLabel', strread(num2str([iFrame:50:iFrame+n_frames_display]),'%s'))
    ylabel('Rhyth. mean')
    xlabel('Frame')
    
    F = getframe(hf);
    aviobj = addframe(aviobj,F);
end

close(hf);
aviobj = close(aviobj);

% save mp3
fln = fullfile('~/diss/data/beats/boeck', [fname, '.wav']);
[y, fs] = wavread(fln );
y = y(1:size(anns, 1)*frame_length*fs);
wavwrite(y, fs, 'test.wav');
system(['lame test.wav ', fullfile(path, [fname, '.mp3'])]);
fprintf('Saved cut mp3 to %s\n', fullfile(path, [fname, '.mp3']));

end

    function [m, n, r] = backtrace(iFrame, best_state, psi_mat, minState, M, N, R)
                % Backtracing
                bestpath = zeros(iFrame,1);
                bestpath(iFrame) = best_state(iFrame);
                for iF=iFrame-1:-1:1
                    bestpath(iF) = psi_mat(bestpath(iF+1),iF+1);
                end

                % add state offset
                bestpath = bestpath + minState - 1;
                [m, n, r] = ind2sub([M, N, R], bestpath);
    end
