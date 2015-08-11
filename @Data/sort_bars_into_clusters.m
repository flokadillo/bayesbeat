function [] = sort_bars_into_clusters(obj, data_per_bar)
% [] = sort_bars_into_clusters(obj, data_per_bar)
%   Sort the features according to clusters
% ----------------------------------------------------------------------
%INPUT parameter:
% data_per_bar    : feature values organised by bar, bar position and
%                   feature dimension: cell(n_bars, num_bar_positions, 
%                   feature_dimensions)
%
%OUTPUT parameter:
% obj.features_organised  : cell(n_files, n_clusters, 
%                   num_bar_positions, feature_dimensions)
%
% 10.08.2015 by Florian Krebs
% ----------------------------------------------------------------------
nFiles = length(obj.file_list);
[~, num_cells, ~] = size(data_per_bar);
features_organised = cell(nFiles, obj.clustering.n_clusters, num_cells, ...
    obj.feature.feat_dim);
nchar=0;
for iFile=1:nFiles
    fprintf(repmat('\b', 1, nchar));
    nchar = fprintf('      %i/%i\n',iFile, nFiles );
    barIds = find(obj.bar2file == iFile);
    for iBar = barIds(:)'
        for iPos = 1:num_cells
            for iDim = 1:obj.feature.feat_dim
                features_organised{iFile, obj.clustering.bar2cluster(iBar), ...
                    iPos, iDim} = [features_organised{iFile, ...
                    obj.clustering.bar2cluster(iBar), iPos, iDim}; ...
                    data_per_bar{iBar, iPos, iDim}];
            end
        end
    end
end
obj.features_organised = features_organised;
end