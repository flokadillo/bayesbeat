function [onset_feat, fr] = python_sprflx(fln, save_it)
% [onset_feat, fr] = robot_bt_feature(fln)
% onset feature for robo beat tracker
% ----------------------------------------------------------------------
% INPUT Parameter:
%   fln               : wave file
%
% OUTPUT Parameter:
%   onset_feat        : [nFrames x 1] onset feature
%   fr                : sampling rate of feature
%
% 19.12.2013 by Florian Krebs
% ----------------------------------------------------------------------
% read wav file
if exist('audioread', 'file')
    [x, fs] = audioread(fln);
else
    [x, fs] = wavread(fln);
end
touched = 0;
% convert to 44.1 kHz
if fs ~= 44100
    x = resample(x, 44100, fs);
    fs = 44100;
    touched = 1;
end
% convert to mono
if size(x, 2) == 2
    x = x(:, 1); % mic = channel 1, pickup = channel 2
    touched = 1;
end
if touched, audiowrite(fln, x, fs); end
fr = 50;
setenv('PYTHONPATH', '/home/florian/diss/src/python/madmom'); % set env path (PYTHONPATH) for this session
[status, onset_feat] = system(['~/diss/src/python/madmom/bin/SuperFlux.py -s --sep " " --fps ', num2str(fr), ' --max_bins 1 ', fln]);
if status == 0
    onset_feat = str2num(onset_feat);
else
    error('    ERROR Feature.python_sprflx\n');
end
if save_it,
    [fpath, fname, ~] = fileparts(fln);
    if ~exist(fullfile(fpath, 'beat_activations'), 'dir')
        system(['mkdir ', fullfile(fpath, 'beat_activations')]);
    end
    save_fln = fullfile(fpath, 'beat_activations', [fname, '.sprflx']);
    fid=fopen(save_fln,'w');
    fprintf(fid,'FRAMERATE: %.4f\nDIMENSION: %i\n', fr, length(onset_feat));
    fprintf(fid,'%d ',onset_feat');
    fclose(fid);
    fprintf(' - saved to %s\n', save_fln);  % file is saved in ComputeDetFunc
end
end
