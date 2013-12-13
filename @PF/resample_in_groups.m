function [outIndex, outWeights, groups] = resample_in_groups(groups, weights)
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
valid_groups = unique(groups);
weights = weights(:);
n_groups = length(valid_groups);
parts_per_group = linspace(0, length(weights), n_groups+1);
parts_per_group = diff(round(parts_per_group));
parts_per_group(end) = length(weights) - sum(parts_per_group(1:end-1));
p = 1;
outWeights = zeros(size(weights));
outIndex = zeros(size(weights));
groups_old = groups;
groups = zeros(size(weights));
for iG=valid_groups'
    group_i = (groups_old==iG);
    n_parts_in_group = sum(group_i);
    if n_parts_in_group < 1
        continue;
    end
    % do resampling
    parts_per_out_group = parts_per_group(valid_groups==iG);
    w_i_norm = exp(normalizeLogspace(weights(group_i)'));
    outIndex_iG = PF.resampleSystematic( w_i_norm(:), parts_per_out_group );
    temp = find(group_i);
    outIndex(p:p+parts_per_out_group-1) = temp(outIndex_iG);
    % compute "total" weight of each group    
    tot_w_i = logsumexp(weights(group_i), 1);
    % divide total weight among new particles
    outWeights(p:p+parts_per_out_group-1) = tot_w_i - log(parts_per_out_group);
    groups(p:p+parts_per_out_group-1) = iG;
    p = p + parts_per_out_group;
end
% outWeights = outWeights / sum(outWeights);
end