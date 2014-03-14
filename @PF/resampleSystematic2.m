function [ indx ] = resampleSystematic2( W, n_samples )
% W ... cell array of normalised weights [n_groups x 1]
% n_samples ... [1 x n_cluster] number of samples after resampling
Q = cumsum(cell2mat(W));
n_particles = sum(n_samples);
T = linspace(0, length(n_samples)-length(n_samples)/n_particles, n_particles) + rand(1)*length(n_samples)/n_particles;
T(n_particles+1) = length(n_samples);
[~, indx] = histc(T, [0; Q]);
indx = indx(1:end-1);