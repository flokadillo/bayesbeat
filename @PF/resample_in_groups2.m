function [outIndex, outWeights, groups_new] = resample_in_groups2(groups, weights, n_max_clusters, warp_fun)
%  [outIndex] = resample_in_groups(groups, weights)
%  resample particles in groups separately
% ----------------------------------------------------------------------
%INPUT parameter:
% groups          : filename (e.g., Media-105907(0.0-10.0).beats)
% weight          : logarithmic weights
%
%
%OUTPUT parameter:
% outIndex        : new resampled indices
%
% 25.09.2012 by Florian Krebs
% ----------------------------------------------------------------------
weights = weights(:);
% group weights according to groups
w_per_group = accumarray(groups, weights, [], @(x) {x});
% sum weights of each group in the log domain
tot_w = cellfun(@(x) logsumexp(x, 1), w_per_group);
% check for groups with zero weights (log(w)=-inf) and remove those
if sum(isinf(tot_w)) > 0
    fprintf('Warning, group with zero weights detected!\n');
    bad_groups = find(isinf(tot_w));
else
    bad_groups = [];
end
% kill cluster with lowest weight
if length(tot_w) - length(bad_groups) > n_max_clusters
    [~, groups_sorted] = sort(tot_w, 'descend');
    bad_groups = unique([bad_groups; groups_sorted(n_max_clusters+1:end)]);
%     bad_groups = unique([bad_groups; groups_sorted(end)]);
end


n_groups = length(tot_w) - length(bad_groups);
parts_per_group = diff(round(linspace(0, length(weights), n_groups+1)));
parts_per_group(end) = length(weights) - sum(parts_per_group(1:end-1));
w_norm = exp(weights - tot_w(groups)); % subtract in log domain and convert to linear
if exist('warp_fun', 'var')
    % do warping
    w_warped = warp_fun(w_norm);
    % normalize weights before resampling
    sum_warped_per_group = accumarray(groups, w_warped);
    w_warped_norm = w_warped ./ sum_warped_per_group(groups); % subtract in log domain and convert to linear
    w_warped_per_group = accumarray(groups, w_warped_norm, [], @(x) {x});
    % resample
    w_warped_per_group(bad_groups) = [];
    outIndex = PF.resampleSystematic2( w_warped_per_group, parts_per_group);
    id_per_group = accumarray(groups, (1:length(weights))', [], @(x) {x});
    id_per_group(bad_groups) = [];
    a = cell2mat(id_per_group);
    outIndex = a(outIndex);
    groups_new = groups(outIndex);
    % do unwarping
    w_fac = w_norm ./ w_warped;
    norm_fac = accumarray(groups_new, w_fac(outIndex));
    outWeights = log(w_fac(outIndex)) + tot_w(groups_new) - log(norm_fac(groups_new));
else
    error('todo...\n')
    outIndex_iG = PF.resampleSystematic( w_i_norm(:), parts_per_out_group );
    % divide total weight among new particles
    outWeights(p:p+parts_per_out_group-1) = tot_w_i - log(parts_per_out_group);
end

end