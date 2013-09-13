function [DetFunc, fr] = Compute_LogFiltSpecFlux(fln, save_it, param)
% [DetFunc, fr] = ComputeDetFunc(fname, results(last_method_id).algorithm)
%   Compute Onset Detection Function
% ----------------------------------------------------------------------
%INPUT parameter:
% fln           : filename of wav (e.g., Media-105907(0.0-10.0).wav)
% oss_type      : e.g., 'SF.filtered82.log'
% param         .lambda (factor for STFT, )
%               .normalize
%               .red
%
%OUTPUT parameter:
% DetFunc        : activations x 1
% fr             : framerate of DetFunc
%
% 11.1.2012 by Florian Krebs
% ----------------------------------------------------------------------

if ~exist(fln,'file')
    fprintf('%s not found\n',fln);
    DetFunc = []; fr = [];
    return
end

% Load audio file
% ----------------------------------------------------------------------
[x, fs] = wavread(fln);
if fs ~= 44100
    x = resample(x, 44100, fs);
    fs = 44100;
end
if size(x, 2) == 2
   % convert to mono
   x = mean(x, 2); 
end
% STFT parameter
% ----------------------------------------------------------------------
param.fftsize = 2048;
param.hopsize = 441; % 10 ms -> fr = 100
winsize = param.fftsize - 1;
param.norm = 1; 
type = 0; % 1=complex, use smaller windows at beginning and end
online = 0; % 1=time frame corresponds to right part of window
[S, t, f] = Feature.STFT(x, winsize, param.hopsize, param.fftsize, fs, type, online, 0, param.norm);
S = S'; % make S = [N x K]
fr = 1 / mean(diff(t));
magnitude = abs(S);

% reduce to 82 bands
load('filterbank82_sb.mat'); % load fb [1024x82]
magnitude = magnitude * fb;

% extract specified frequency bands
if isfield(param, 'min_f')
    
    % find corresponding frequency bin in Hz
    [~, min_ind] = min(abs(f - param.min_f));
    [~, max_ind] = min(abs(f - param.max_f));
    
    % find corresponding frequency band of the filterbank
    min_ind = find(fb(max([min_ind, 2]), :));
    % fb only uses frequencies up to 16.75 kHz (bin 778)
    [~, max_ind] = max(fb(min([778, max_ind]), :));
    
    magnitude = magnitude(:, min_ind:max_ind);
end

% logarithmic amplitude
param.lambda = 10221;
magnitude = log10(param.lambda * magnitude + 1);

% compute flux
difforder = 1;
Sdif = magnitude - [magnitude(1:difforder,:); magnitude(1:end-difforder,:)]; % Sdif(n) = S(n) - S(n-difforder);

% halfway rectifying
Sdif = (Sdif+abs(Sdif)) / 2;    
% DetFunc = Sdif;
% sum over frequency bands
DetFunc = sum(Sdif,2);

if save_it
    [pathstr,fname,~] = fileparts(fln);
    if isfield(param,'save_path')
        save_fln = fullfile(param.save_path,[fname,'.onsets.',oss_type]);
    else
        save_fln = fullfile(pathstr,[fname,'.onsets.',oss_type]);
    end
    fid=fopen(save_fln,'w');
    fprintf(fid,'FRAMERATE: %.4f\nDIMENSION: %i\n', fr, length(DetFunc));
    fprintf(fid,'%d ',DetFunc');
    fclose(fid);
end

end

