function [] = sort_bars_into_clusters(obj, dataPerBar)
% [dataPerFile] = sort_bars_into_clusters(dataPerBar, clusterIdx, bar2file)
%   Sorts dataPerBar according to clusters
% ----------------------------------------------------------------------
%INPUT parameter:
% dataPerBar      :
% clusterIdx      :
%
% bar2file        :
%
%OUTPUT parameter:
% dataPerFile       : cell(n_files, n_clusters, bar_grid_max, featureDim)
%
% 26.7.2012 by Florian Krebs
% ----------------------------------------------------------------------
nFiles = length(obj.file_list);
[~, num_cells, ~] = size(dataPerBar);
features_organised = cell(nFiles, obj.n_clusters, num_cells, ...
    obj.feature.feat_dim);
nchar=0;
for iFile=1:nFiles
    fprintf(repmat('\b', 1, nchar));
    nchar = fprintf('      %i/%i\n',iFile, nFiles );
    barIds = find(obj.bar2file == iFile);
    for iBar = barIds(:)'
        for iPos = 1:num_cells
            for iDim = 1:obj.feature.feat_dim
                features_organised{iFile, obj.bar2cluster(iBar), ...
                    iPos, iDim} = [features_organised{iFile, ...
                    obj.bar2cluster(iBar), iPos, iDim}; ...
                    dataPerBar{iBar, iPos, iDim}];
            end
        end
    end
end
obj.features_organised = features_organised;
end