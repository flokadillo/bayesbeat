function [ indx ] = resampleSystematic( w, n_samples )
% n_samples ... number of samples after resampling 
w = w(:);
if sum(w) > eps
    w = w / sum(w);
else
    w = ones(size(w)) / length(w); 
end
if exist('n_samples', 'var')
    N = n_samples;
else
    N = length(w);
end
Q = cumsum(w);

T = linspace(0,1-1/N,N) + rand(1)/N;
%  T = linspace(0,1-1/N,N);
T(N+1) = 1;

[~, indx] = histc(T, [0; Q]);
indx = indx(1:end-1);