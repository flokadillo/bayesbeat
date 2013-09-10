function [Output] = extract_bars_from_feature(source, featExt, barGrid, frame_length, dooutput)
% [out3, out4] = Analyze_Onset_Strength_Inside_A_Bar(source, featExt, mode2 [,
% numbins, saveflag, bin_thresh] )
%   Maps the time t to a bar position
% ----------------------------------------------------------------------
%INPUT parameter:
% source            : folder, filename or filelist
% featExt           : extension of feature file (e.g., BoeckNN,
%                       bt.SF.filtered82.log). It is assumed that the
%                       feature file is located in a subfolder
%                       "beat_activations" within the input source
%                       folder
% barGrid           : number of gridpoints of one bar (4/4 meter) [64]
% frame_length       : feature frame_length [0.02]
%
%OUTPUT parameter:
% Output.dataPerBar : [nBars x barGrid] matrix of double 
%                       from this, cellfun(@mean, Output.dataPerBar) computed
%                       the mean of each bar and position
% Output.bar2file   : [1 x nBars] vector
% Output.fileNames  : [nFiles x 1] vector
%
% 26.7.2012 by Florian Krebs
% ----------------------------------------------------------------------

% ####################### PARAMETERS ######################################
% set parameters
addpath('~/diss/src/matlab/libs/matlab_utils');
if nargin == 2, 
    barGrid = 64;
    frame_length = 0.02;
    dooutput = 0; 
end
[listing, nFiles] = parseSource(source, featExt);
% bar grid for triple and duple meter
Output.dataPerBar = []; idLastBar = 0;
Output.fileNames = cell(nFiles, 1);
barGrid_max = barGrid;
fprintf('    Organize feature values (%s) into bars ...\n', featExt);
%main loop over all files
for iFile=1:nFiles
    [dataPath, fname, ~] = fileparts(listing(iFile).name);
    fprintf('      %i/%i) %s \n', iFile, nFiles, fname);
    [ annots, error ] = loadAnnotations( dataPath, fname, 'wmb', dooutput );
    if error,
        if dooutput, fprintf('Skipping %s\n',fname); end
        continue;
%     elseif ~ismember(annots.meter, [3; 4])
%         if dooutput, fprintf('Skipping %s because of meter (%i)\n',fname, annots.meter); end
%         continue;
    end
    Output.fileNames{iFile} = listing(iFile).name;
    featureFln = fullfile(dataPath, 'beat_activations', [fname, '.', featExt]);
    % for one bar to be extracted, the downbeat of the bar itself and the
    % next downbeat has to be present in the annotations. Otherwise, it is
    % discarded
    b1 = annots.beats(round(rem(annots.beats(:,2),1)*10) == 1,1);
    if length(b1) > 1
        % effective number of bar positions (depending on meter)
        if length(annots.meter) == 1
            barGridEff = barGrid*annots.meter/4;
        else
            barGridEff = ceil(barGrid*annots.meter(1)/annots.meter(2));
        end
        % collect feature values and determine the corresponding position
        % in a bar
        barData = get_feature_at_bar_grid(featureFln, annots.beats, barGrid, barGridEff, frame_length);
        if ~isempty(barData)
            [nNewBars, currBarGrid] = size(barData);
            % for triple meter fill in empty cells
            if currBarGrid < barGrid_max,
                barData = [barData, cell(size(barData, 1), barGrid_max-size(barData, 2))];
            elseif currBarGrid > barGrid,
                barGrid_max = currBarGrid;
            end
            Output.dataPerBar = [Output.dataPerBar; barData];          
            Output.bar2file((idLastBar + 1):(idLastBar + nNewBars)) = iFile;
            idLastBar = idLastBar + nNewBars;
        end
    else
        if dooutput, fprintf('    %s contains only one bar -> skip it\n', fname); end
    end
end

end % FUNCTION end


% load feature values from file and up/downsample to frame_length
function [DetFunc, fr] = load_features(featureFln, frame_length)

if exist(featureFln,'file')
    [DetFunc, fr] = readActivations(featureFln);
else
    error('Analyze_Bars: don''t know how to compute %s\n', featureFln);
    return
end

%make column vector
DetFunc = DetFunc(:);

if ~isempty(frame_length) && (abs(1/fr - frame_length) > 0.001)
    % adjusting framerate:
    DetFunc = Feature.change_frame_rate( DetFunc, fr, 1/frame_length );
    fr = 1/frame_length;
end

end


function [barData] = get_feature_at_bar_grid(featureFln, beats, barGrid, barGridEff, frame_length)
% barData   [nBars x barGrid] cell array features values per bar and bargrid

% load feature values from file and up/downsample to frame_length
[E, fr] = load_features(featureFln, frame_length);

% if feature vector to short for annotations, copy last value
if length(E)/fr < beats(end, 1), E = [E; E(end)]; end

if beats(end, 1) > length(E) * frame_length;
    fprintf('   Warning: beat annotations longer than audio file\n');
    beats = beats(beats(:, 1) <= length(E) * frame_length, :);
end

[nBars, ~, barStartIdx] = Data.get_full_bars(beats);
btype = round(rem(beats(:,2),1)*10);
meter = max(btype);


if ismember(meter, [3, 4])
    beatsBarPos = ((0:meter) * barGrid / 4) + 1;
elseif ismember(meter, [8, 9])
    beatsBarPos = ((0:meter) * barGrid / 8) + 1;
else 
    error('meter %i unknown\n', meter);
end

barData = cell(nBars, barGridEff);



for iBar=1:nBars
	% compute start and end frame of bar using fr
	startFrame = floor(beats(barStartIdx(iBar), 1) * fr) + 1; % first frame of bar
	endFrame = floor(beats(barStartIdx(iBar)+meter, 1) * fr) + 1; % first frame of next bar

	% extract feature for this bar
	featBar = E(startFrame:endFrame);

	% set up time frames of bar
	t = (startFrame:endFrame) / fr - 1/(2*fr); % subtract half frame (1/(2*fr)) to yield center of frame

	% interpolate to find bar position of each audio frame
	barPosLin = round(interp1(beats(barStartIdx(iBar):barStartIdx(iBar)+meter, 1), beatsBarPos, t,'linear','extrap'));
    barPosLin(barPosLin < 1) = 1;
    
	% group all feature values that belong to the same barPos
	currBarData = accumarray(barPosLin', featBar(:), [], @(x) {x});
    
	% add to bar cell array	
	barData(iBar, :) = currBarData(1:barGridEff); % last element belongs to next bar -> remove it
end

% interpolate missing values
if sum(sum(cellfun(@isempty, barData))) > 0
    data_means = cellfun(@mean, barData);
    [rows, cols] = size(barData);
    times = 1:cols;
    for i = 1:rows
        missing_data = isnan(data_means(i, :));
        if sum(missing_data) > 0
            barData(i, missing_data) = num2cell(interp1(times(~missing_data), ...
                data_means(i, ~missing_data), times(missing_data), 'linear', 'extrap'));
        end
    end
end

end

function barData = add2BarMatrix(E, barpos)
% barData: for each bar find mean of odf for each bar postition

num_complete_bars = 0;
nPos = max(barpos);

% only use complete bars:
start_id = find(barpos(1:nPos) == 1, 1);
if isempty(start_id)
    start_id = find(diff(barpos) < 0, 1, 'first') + 1;
end
end_id = length(barpos)-1;
firstDownbeatHasPassed = 1;

if end_id > start_id
    groupid = start_id(1);
    for i=start_id(1):end_id
        % two features at the same bar position form a group
        if barpos(i) == barpos(i+1)
            groupid = [groupid i+1];
        else
            
            % save old position
            % barData(num_complete_bars+1, barpos(i)) = mean(E( groupid));
            barData{num_complete_bars+1, barpos(i)} = E( groupid);
            groupid = i+1;
            
            
            if barpos(i+1) > barpos(i)
                steps = barpos(i):barpos(i+1);
            else
                steps = [barpos(i):nPos, 1:barpos(i+1)];
            end
            
            % jumps in barposition -> interpolate
            if length(steps) > 2
                for iStep=1:length(steps)-2
                    indPos = find(steps==nPos);
                    if isempty(indPos)
                        barData{num_complete_bars+1, steps(iStep+1)} = mean(E(groupid-1:groupid));
                    else
                        if iStep+1 <= indPos
                            barData{num_complete_bars+1, steps(iStep+1)} = mean(E(groupid-1:groupid));
                        else
                            barData{num_complete_bars+2, steps(iStep+1)} = mean(E(groupid-1:groupid));
                        end
                        
                    end
                end
            end
            
            if (barpos(i+1) < barpos(i)) && firstDownbeatHasPassed % new bar
                num_complete_bars = num_complete_bars + 1;
            end
        end
    end
    if isempty(barData{1, 1})
        barData{1} = barData{1, 2};
    end
else % no complete bar in the signal
    barData = [];
end
barData = barData(1:num_complete_bars, :);
end

function [listing, nFiles] = parseSource(source, featExt)

% parse source
if iscell(source)  % cell array with path and filenames
    nFiles = length(source);
    listing = cell2struct(source, 'name', nFiles);
else % name of file or folder
    listing = dir(source);
    if length(listing) > 1 % folder
        fileList = dir(fullfile(source, 'beat_activations', ['*.', featExt]));
        if isempty(fileList)
            fprintf('ERROR extract_bars_from_feature: No feature files in %s found \n', source);
            return;
        end
        nFiles = length(fileList);
        clear listing;
        for iFile=1:nFiles
            fname = strrep(fileList(iFile).name, featExt, 'wav');
            listing(iFile).name = fullfile(source, fname);
        end
        
    else
        listing(1).name = source;
        nFiles = 1;
    end
end
end
