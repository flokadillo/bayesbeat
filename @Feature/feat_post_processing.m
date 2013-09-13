function [ DetFunc, fr ] = feat_post_processing( wavFileName, param )
feat_post_processing.m
%[ DetFunc, fr ] = compute_Bt_Log_Filt_Spec( wavFileName )
%  computes beat detection function based on:
%    Böck, S., Krebs, F. and Schedl, M. (2012). Evaluating the Online Capabilities of Onset Detection Methods.
%    (ISMIR 2012)
%
% ------------------------------------------------------------------------
%INPUT parameters:
% wavFileName           : input WAV file
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

% PARAMETERS:
% ===========

param.offline = 1;
param.logThresh = 30;           % Mean + 1.7 * Variance of all feature values
param.normalizingConst = 35;    % Mean + 2.0 * Variance of all feature values

save_it = 0;

[DetFunc, fr] = Compute_LogFiltSpecFlux(wavFileName, save_it, param); % compute and save
DetFunc = DetFunc * param.amplitudeFactor ;

if param.moving
    y = moving(DetFunc, 77);
end

if param.doMvavg
    % moving average
    dm = mvavg(DetFunc, 100, 'normal');
    if ~isempty(dm)
        DetFunc = DetFunc-[0; dm(1:end-1)];
%         DetFunc(DetFunc<0) = 0;
    end
end

if param.compress
    % compress
    DetFunc(DetFunc>param.logThresh) = param.logThresh + log(DetFunc(DetFunc>param.logThresh) - param.logThresh + 1);
    % make it [0..1]
    DetFunc = DetFunc / param.normalizingConst;
end

if param.norm_each_file == 2
    DetFunc = DetFunc - mean(DetFunc);
    DetFunc = DetFunc / var(DetFunc);
elseif param.norm_each_file == 1
    DetFunc = DetFunc - min(DetFunc) + 0.001;
    DetFunc = DetFunc / max(DetFunc);
end


end

