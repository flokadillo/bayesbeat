function [outIndex, outWeights] = resample_in_groups(groups, weights)
%  [outIndex] = resample_in_groups(groups, weights)
%  resample particles in groups separately
% ----------------------------------------------------------------------
%INPUT parameter:
% groups          : filename (e.g., Media-105907(0.0-10.0).beats)
%
%OUTPUT parameter:
% bt_act        : activations x 1
%
% 25.09.2012 by Florian Krebs
% ----------------------------------------------------------------------
parts_per_group = linspace(0, length(weights), max(groups)+1);
parts_per_group = diff(round(parts_per_group));
parts_per_group(end) = length(weights) - sum(parts_per_group(1:end-1));
n_groups = max(groups);
p = 1;
outWeights = zeros(size(weights));
outIndex = zeros(size(weights));
for iG=1:n_groups
    group_i = (groups==iG);
    n_parts_in_group = sum(group_i);
    if n_parts_in_group < 1
        continue;
    end
    w = sum(weights(group_i)) / n_parts_in_group;
    
    % do resampling
    outIndex_iG = PF.resampleSystematic( weights(group_i), parts_per_group(iG) );
    temp = find(group_i);
    outIndex(p:p+parts_per_group(iG)-1) = temp(outIndex_iG);
    outWeights(p:p+parts_per_group(iG)-1) = w/length(outIndex_iG);
    p = p + parts_per_group(iG);
end
outWeights = outWeights / sum(outWeights);
end