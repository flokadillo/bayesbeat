function r = logsumexp(X,dim)
%LOG_SUM_EXP Numerically stable computation of log(sum(exp(X), dim))
% [r] = log_sum_exp(X, dim)
%
% Inputs :
%
% X : Array
%
% dim : Sum Dimension <default = 1>, means summing over the columns
% This file is taken from pmtk3.googlecode.com

maxval = max(X,[],dim);
sizes = size(X);
if dim == 1
    normmat = repmat(maxval,sizes(1),1);
else
    normmat = repmat(maxval,1,sizes(2));
end

r = maxval + log(sum(exp(X-normmat),dim));





