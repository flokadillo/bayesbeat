function [y, L] = normalizeLogspace2D(x)
% x is a matrix
% Normalize in logspace while avoiding numerical underflow
% y(i,j) = x(i,j) - logsumexp(x(:))
% eg [logPost, L] = normalizeLogspace(logprior + loglik)
% post = exp(logPost);
%%
% This file is modified from pmtk3.googlecode.com
L = logsumexp(x(:));
%y = x - repmat(L, 1, size(x,2));
y = x - L;
end