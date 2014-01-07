function [dataPerFile] = sort_bars_into_clusters(dataPerBar, clusterIdx, bar2file)
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
nFiles = max(bar2file);
nClusters = max(clusterIdx);
[~, barGrid] = size(dataPerBar);
dataPerFile = cell(nFiles, nClusters, barGrid);
for iFile=1:nFiles
    fprintf('      %i/%i\n',iFile, nFiles );
    barIds = find(bar2file == iFile); 
        for iBar = 1:length(barIds)  
            if iBar==48 && iFile==97
                klj=987;
            end
            for iPos = 1:barGrid
%                 iPos
                dataPerFile{iFile, clusterIdx(barIds(iBar)), iPos} = ...
                    [dataPerFile{iFile, clusterIdx(barIds(iBar)), iPos}; dataPerBar{barIds(iBar), iPos}];
            end
        end
end




