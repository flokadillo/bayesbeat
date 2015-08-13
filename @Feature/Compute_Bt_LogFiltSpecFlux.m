function [ DetFunc, fr ] = Compute_Bt_LogFiltSpecFlux( fln, param )
%[ DetFunc, fr ] = compute_Bt_Log_Filt_Spec( fln )
%  computes beat detection function based on:
%    Böck, S., Krebs, F. and Schedl, M. (2012). Evaluating the Online Capabilities of Onset Detection Methods.
%    (ISMIR 2012)
%
% ------------------------------------------------------------------------
%INPUT parameters:
% fln           : input WAV file
% param
%
%OUTPUT parameters:
% DetFunc               : beat detection function
% fr                    : framerate of DetFunc
%
%DEPENDS on:
% ComputeDetFunc.m
% mvavg.m
%
% 06.09.2012 by Florian Krebs
% ------------------------------------------------------------------------
if ~exist(fln,'file')
    fprintf('%s not found\n',fln);
    DetFunc = []; fr = [];
    return
end
% Set default parameters:
% ===========
if ~isfield(param, 'offline')
    param.offline = 1;
end
if ~isfield(param, 'min_f')
    param.min_f = 0;
end
if ~isfield(param, 'max_f')
    param.max_f = 22050;
end
if ~isfield(param, 'frame_length')
    param.frame_length = 0.02;
end
if ~isfield(param, 'fftsize')
    param.fftsize = 2048;
end
if ~isfield(param, 'hopsize')
    param.hopsize = 441;
end
if ~isfield(param, 'lambda')
    param.lambda = 100000;
end
if ~isfield(param, 'norm_each_file')
    param.norm_each_file = 0;
end
if ~isfield(param, 'feat_type')
   param.feat_type = 'superflux';
end
if ~isfield(param, 'doMvavg')
   param.doMvavg = 0;
end
if ~isfield(param, 'save_it')
   param.save_it = 0;
end

if param.min_f == 0 && param.max_f > 40000 % use Sebastian's superflux
    [~, home] = system('echo $HOME');
    fln = strrep(fln, '~', home(1:end-1)); % remove \n at the end of home
    % set env path (PYTHONPATH) for this session
    fr = round(1/param.frame_length);
    setenv('PYTHONPATH', '/home/florian/diss/src/python/madmom'); 
    cmd = ['~/diss/src/python/madmom/bin/SuperFlux.py --save ', ...
        '--sep " " --fps ', num2str(fr), ' --max_bins 1 ', ...
        'single "', fln, '"'];
    [status, DetFunc] = system(cmd);
    % remove leading header from string
    delim_pos = strfind(DetFunc, ' ');
    DetFunc = str2num(DetFunc(delim_pos(2)+1:end));
    if status ~= 0 || isempty(DetFunc)
        error('Could not extract features from %s\n', fln);
    end
else
    % Load audio file
    % ----------------------------------------------------------------------
    if exist('audioread', 'file')
        [x, fs] = audioread(fln);
    else
        [~, ~, ext] = fileparts(fln);
        if strcmp(ext, '.flac')
            fprintf('    Converting flac to wav ...\n');
            if ~exist(strrep(fln, 'flac', 'wav'), 'file')
                fprintf('    Converting flac to wav ...\n');
                system(['flac -df --output-name="', ...
                    strrep(strrep(fln, 'flac', 'wav'), '~', '${HOME}'), ...
                    '" "', strrep(fln, '~', '${HOME}'), '"']);
            end
            fln = strrep(fln, 'flac', 'wav');
        end
        [x, fs] = wavread(fln);
    end
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
    winsize = param.fftsize - 1;
    type = 0; % 1=complex, use smaller windows at beginning and end
    online = 0; % 1=time frame corresponds to right part of window
    [S, t, f] = Feature.STFT(x, winsize, param.hopsize, param.fftsize, fs, type, online, 0);
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
    magnitude = log10(param.lambda * magnitude + 1);
    % compute flux
    difforder = 1;
    Sdif = magnitude - [magnitude(1:difforder,:); magnitude(1:end-difforder,:)]; % Sdif(n) = S(n) - S(n-difforder);
    % halfway rectifying
    Sdif = (Sdif+abs(Sdif)) / 2;
    % sum over frequency bands
    DetFunc = sum(Sdif,2);
end

if param.doMvavg
    % moving average
    dm = Feature.mvavg(DetFunc, 100, 'normal');
    if ~isempty(dm)
        DetFunc = DetFunc-[0; dm(1:end-1)];
    end
end
if param.norm_each_file == 2
    % zero mean and unit variance
    DetFunc = DetFunc - mean(DetFunc);
    DetFunc = DetFunc / var(DetFunc);
elseif param.norm_each_file == 1
    DetFunc = DetFunc - min(DetFunc) + 0.001;
    DetFunc = DetFunc / max(DetFunc);
end
% adjust framerate of features
if abs(1/fr - param.frame_length) > 0.001
    DetFunc = Feature.change_frame_rate(DetFunc, ...
        round(1000*fr)/1000, 1/param.frame_length );
    fr = 1/param.frame_length;
end
DetFunc = DetFunc(:);
if param.save_it
    [pathstr,fname,~] = fileparts(fln);
    fname = [fname, '.', param.feat_type];
    if isfield(param,'save_path')
        pathstr = param.save_path;
    else
        pathstr = fullfile(pathstr, 'beat_activations');
    end
    if ~exist(pathstr, 'dir')
        system(['mkdir ', pathstr]);
    end
    save_fln = fullfile(pathstr, fname);
    fid=fopen(save_fln,'w');
    fprintf(fid,'FRAMERATE: %.4f\nDIMENSION: %i\n', fr, length(DetFunc));
    fprintf(fid,'%d ',DetFunc');
    fclose(fid);
    fprintf('    Feature saved to %s\n', save_fln);
end

end

