function [ out ] = mvavg( signal, winsize, type )
%MVAVG Computes moving filters
% example for moving average filter:
% out(k) = mean(signal(k-winsize+1:k))
% --------------------------------------------------------------------
if winsize > length(signal)
    fprintf('ERROR: signal too short !\n');
    out = [];
    return
end
if strcmp(type,'normal')
    % construct kernel
    kernel = ones(winsize,1);
    kernel = kernel / (sum(kernel));
    % convolution
    if size(signal,1) == 1, signal = signal'; end
    out = conv(kernel, signal);
    out = out(1:length(signal));
    % set beginning to the mean
    out(1:winsize-1) = mean(signal(1:winsize-1));
elseif strcmp(type,'halfmean')
    % construct kernel
    kernel = ones(winsize,1);
    kernel = kernel / (2*sum(kernel));
    % convolution
    if size(signal,1) == 1, signal = signal'; end
    out = conv(kernel, signal);
    out = out(1:length(signal));
    % set beginning to the mean
    out(1:winsize-1) = mean(signal(1:winsize-1))/2;
    
end
end

