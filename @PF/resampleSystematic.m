function [ indx ] = resampleSystematic( w )

N = length(w);
Q = cumsum(w);

T = linspace(0,1-1/N,N) + rand(1)/N;
T(N+1) = 1;

[~, indx] = histc(T, [0; Q]);
indx = indx(1:end-1);