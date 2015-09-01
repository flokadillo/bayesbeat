function [dataPerFile] = sort_patts_into_clusters(dataPerPatt, clusterIdx, patt2file)
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
% dataPerFile       : dataPerFile(fileId, clusterId, barPos)
%
% 26.7.2012 by Florian Krebs
% ----------------------------------------------------------------------
nFiles = max(patt2file);
nClusters = max(clusterIdx);
[~, barGrid] = size(dataPerPatt);
dataPerFile = cell(nFiles, nClusters, barGrid);
nchar=0;
for iFile=1:nFiles
    fprintf(repmat('\b', 1, nchar));
    nchar = fprintf('      %i/%i\n',iFile, nFiles );
    barIds = find(patt2file == iFile); 
        for iBar = 1:length(barIds) 
            for iPos = 1:barGrid
                dataPerFile{iFile, clusterIdx(barIds(iBar)), iPos} = ...
                    [dataPerFile{iFile, clusterIdx(barIds(iBar)), iPos}; dataPerPatt{barIds(iBar), iPos}];
            end
        end
end