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
if sum(isnan(tot_w)) > 0
    bad_groups = find(isnan(tot_w));
    fprintf('   %i of %i groups have zero weights and are removed!\n', length(bad_groups), length(tot_w));
else
    bad_groups = [];
end
% kill clusters with lowest weight to prevent more than n_max_clusters
% clusters
if length(tot_w) - length(bad_groups) > n_max_clusters
    [~, groups_sorted] = sort(tot_w, 'descend');
    fprintf('    too many groups (%i)! -> removing %i\n', length(tot_w) - length(bad_groups), ...
        length(tot_w) - length(bad_groups) - n_max_clusters);
    bad_groups = unique([bad_groups; groups_sorted(n_max_clusters+1:end)]);
end

id_per_group = accumarray(groups, (1:length(weights))', [], @(x) {x});
id_per_group(bad_groups) = [];
id_per_group = cell2mat(id_per_group);

n_groups = length(tot_w) - length(bad_groups);
% cumulative sum of particles per group. Each group should have an
% approximative equal number of particles.
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
    outIndex = id_per_group(outIndex);
    groups_new = groups(outIndex);
    % do unwarping
    w_fac = w_norm ./ w_warped;
    norm_fac = accumarray(groups_new, w_fac(outIndex));
    outWeights = log(w_fac(outIndex)) + tot_w(groups_new) - log(norm_fac(groups_new));
else
    w_norm_per_group = accumarray(groups, w_norm, [], @(x) {x});
    w_norm_per_group(bad_groups) = [];
    outIndex = PF.resampleSystematic2(w_norm_per_group, parts_per_group);
    outIndex = id_per_group(outIndex);
    groups_new = groups(outIndex);
    % divide total weight among new particles
    outWeights = tot_w(groups_new) - log(mean(parts_per_group));
end
end
