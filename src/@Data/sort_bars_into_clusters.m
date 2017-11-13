function [features_by_clusters] = sort_bars_into_clusters(obj)
% [] = sort_bars_into_clusters(obj, data_per_bar)
%   Sort the features according to clusters
% ----------------------------------------------------------------------
%INPUT parameter:
% data_per_bar    : feature values organised by bar, bar position and
%                   feature dimension: cell(n_bars, num_bar_positions, 
%                   feature_dimensions)
%
%OUTPUT parameter:
% features_by_clusters  : cell(n_clusters, num_bar_positions, feature_dimensions)
%
% 10.08.2015 by Florian Krebs
% ----------------------------------------------------------------------
[~, num_cells, ~] = size(obj.features_per_bar);
features_by_clusters = cell(obj.clustering.n_clusters, num_cells, obj.feature.feat_dim);
nchar=0;
for iPos = 1:num_cells
    for iDim = 1:obj.feature.feat_dim
        for iClust = 1:obj.clustering.n_clusters
            features_by_clusters{iClust,iPos, iDim} = cell2mat(obj.features_per_bar(obj.clustering.bar2cluster==iClust, iPos, iDim));
        end
    end
    fprintf(repmat('\b', 1, nchar));
    nchar = fprintf('      %i/%i\n',iPos, num_cells );
end
% Clear output
fprintf(repmat('\b', 1, nchar));
end