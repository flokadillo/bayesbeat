function [Output] = extract_bars_from_feature(source, feature_type, whole_note_div, frame_length, pattern_size, dooutput)
% [out3, out4] = Analyze_Onset_Strength_Inside_A_Bar(source, feature_type, mode2 [,
% numbins, saveflag, bin_thresh] )
%   Maps the time t to a bar position
% ----------------------------------------------------------------------
%INPUT parameter:
% source            : folder, filename or filelist
% feature_type         : extension of feature file (e.g., BoeckNN,
%                       bt.SF.filtered82.log). It is assumed that the
%                       feature file is located in a subfolder
%                       "beat_activations" within the input source
%                       folder
% whole_note_div     : number of gridpoints of one whole note [64]
% frame_length       : feature frame_length [0.02]
%
%OUTPUT parameter:
% Output.dataPerBar : [nBars x whole_note_div] matrix of double 
%                       from this, cellfun(@mean, Output.dataPerBar)
%                       computes the mean of each bar and position and can
%                       be plotted by plot(mean(cellfun(@mean, Output.dataPerBar)))
% Output.bar2file   : [1 x nBars] vector
% Output.fileNames  : [nFiles x 1] vector
% Output.file2meter : [nFiles x 2]
%
% 26.7.2012 by Florian Krebs
% ----------------------------------------------------------------------

% ####################### PARAMETERS ######################################
% set parameters
addpath('~/diss/src/matlab/libs/matlab_utils');
if nargin == 2, 
    whole_note_div = 64;
    frame_length = 0.02;
    dooutput = 0; 
end
if ~exist('pattern_size', 'var')
    pattern_size = 'bar';
end

[listing, nFiles] = parseSource(source, feature_type);
% bar grid for triple and duple meter
Output.dataPerBar = []; idLastBar = 0;
Output.fileNames = cell(nFiles, 1);
Output.file2meter = zeros(nFiles, 2);
fprintf('    Organize feature values (%s) into bars ...\n', feature_type);

% whole_note_div is loop over all files to find maximum bargrid
if strcmp(pattern_size, 'bar')
    bar_grid_max = 0;
    for iFile=1:nFiles
        [ meter, ~ ] = Data.load_annotations_bt(listing(iFile).name, 'meter');
        if length(meter) == 1
           bar_grid_max = max([bar_grid_max; whole_note_div * meter / 4]);
           Output.file2meter(iFile, 1) = meter;
           Output.file2meter(iFile, 2) = 4;
           fprintf('\n    Warning: meter file has only one value, assuming quarter note beats\n');
        else
           bar_grid_max = max([bar_grid_max; ceil(whole_note_div * meter(1) / meter(2))]);
           Output.file2meter(iFile, :) = meter;
        end
        
    end
else
    bar_grid_max = round(whole_note_div / 4);
end
nchar = 0;
%main loop over all files
for iFile=1:nFiles
    [dataPath, fname, ext] = fileparts(listing(iFile).name);
    fprintf(repmat('\b', 1, nchar));
    nchar = fprintf('      %i/%i) %s', iFile, nFiles, fname);
    [ meter, error1 ] = Data.load_annotations_bt(listing(iFile).name, 'meter');
    [ beats, error2 ] = Data.load_annotations_bt(listing(iFile).name, 'beats');
    if (error1 || error2),
        if dooutput, fprintf('Error loading annotations, skipping %s\n', fname); end
        continue;
%     elseif ~ismember(meter, [3; 4])
%         if dooutput, fprintf('Skipping %s because of meter (%i)\n',fname, meter); end
%         continue;
    end
    Output.fileNames{iFile} = listing(iFile).name;
    wav_fln = listing(iFile).name;
%     featureFln = fullfile(dataPath, 'beat_activations', [fname, '.', feature_type]);
    % for one bar to be extracted, the downbeat of the bar itself and the
    % next downbeat has to be present in the annotations. Otherwise, it is
    % discarded
    b1 = find(beats(:, 3) == 1);
    
    if (length(b1) <= 1) && strcmp(pattern_size, 'bar')
        if dooutput, 
            fprintf(repmat('\b', 1, nchar));
            fprintf('    %s contains only one bar -> skip it\n', fname); 
            nchar = 0;
        end
    else
        % effective number of bar positions (depending on meter)
        if strcmp(pattern_size, 'bar')
            if length(meter) == 1
                bar_grid_eff = whole_note_div * meter / 4;
                meter(2) = 4;
            else
                bar_grid_eff = ceil(whole_note_div * meter(1) / meter(2));
            end
        else
            % beat-length patterns: suppose quarter beats 
            bar_grid_eff = whole_note_div / 4;
        end
        
        % collect feature values and determine the corresponding position
        % in a bar
        [barData, nchar, Output.bar_pos_per_frame{iFile}, Output.pattern_per_frame{iFile}] = ...
            get_feature_at_bar_grid(wav_fln, feature_type, beats, whole_note_div, bar_grid_eff, ...
            frame_length, pattern_size, meter, nchar);
        if ~isempty(barData)
            [nNewBars, currBarGrid] = size(barData);
            % for triple meter fill in empty cells
            if currBarGrid < bar_grid_max,
                barData = [barData, cell(size(barData, 1), bar_grid_max-size(barData, 2))];
            elseif currBarGrid > whole_note_div,
                bar_grid_max = currBarGrid;
            end
            Output.dataPerBar = [Output.dataPerBar; barData];          
            Output.bar2file((idLastBar + 1):(idLastBar + nNewBars)) = iFile;
            Output.pattern_per_frame{iFile} = Output.pattern_per_frame{iFile} + idLastBar;
            idLastBar = idLastBar + nNewBars;
        end
    end
end
fprintf('\n');
end % FUNCTION end


% load feature values from file and up/downsample to frame_length
function [DetFunc, fr] = load_features(wav_fln, feature_type, frame_length)
% points = strfind(wav_fln, '.');
Feat = Feature({feature_type}, frame_length);
DetFunc = Feat.load_feature(wav_fln);
%make column vector
DetFunc = DetFunc(:);
fr = 1/Feat.frame_length;
end


function [barData, nchar, bar_pos_per_frame, pattern_per_frame] = get_feature_at_bar_grid(wav_fln, ...
    feature_type, beats, whole_note_div, bar_grid_eff, frame_length, pattern_size, meter, nchar)
% barData   [nBars x whole_note_div] cell array features values per bar and bargrid

% load feature values from file and up/downsample to frame_length
[E, fr] = load_features(wav_fln, feature_type, frame_length);
nchar = 0;
% if feature vector to short for annotations, copy last value
if length(E)/fr < beats(end, 1), E = [E; E(end)]; end

if beats(end, 1) > length(E) * frame_length;
    fprintf('     Warning: beat annotations longer than audio file\n');
    nchar = 0;
    beats = beats(beats(:, 1) <= length(E) * frame_length, :);
end

if strcmp(pattern_size, 'bar')
    [nBars, ~, barStartIdx] = Data.get_full_bars(beats);

else
    nBars = size(beats, 1) - 1;
    barStartIdx = 1:nBars;
%     btype = ones(size(beats, 1), 1);
    meter = 1;
end
beatsBarPos = ((0:meter(1)) * whole_note_div / meter(2)) + 1;

barData = cell(nBars, bar_grid_eff);
bar_pos_per_frame = nan(size(E), 'single');
pattern_per_frame = nan(size(E), 'single');
for iBar=1:nBars
	% compute start and end frame of bar using fr
	startFrame = max([floor(beats(barStartIdx(iBar), 1) * fr), 1]);  % first frame of bar
	nextFrame = floor(beats(barStartIdx(iBar)+meter(1), 1) * fr); % first frame of next bar

	% extract feature for this bar
	featBar = E(startFrame:nextFrame);

	% set up time frames of bar
	t = (startFrame:nextFrame) / fr - 1/(2*fr); % subtract half frame (1/(2*fr)) to yield center of frame

	% interpolate to find bar position of each audio frame
	barPosLin = round(interp1(beats(barStartIdx(iBar):barStartIdx(iBar)+meter(1), 1), beatsBarPos, t,'linear','extrap'));
    barPosLin(barPosLin < 1) = 1;
    % bar position 64th grid per frame
    bar_pos_per_frame(startFrame:nextFrame-1) = barPosLin(1:end-1);
    pattern_per_frame(startFrame:nextFrame-1) = ones(nextFrame-startFrame, 1) * iBar;
	% group all feature values that belong to the same barPos
	currBarData = accumarray(barPosLin', featBar(:), [bar_grid_eff+1, 1], @(x) {x});
    
	% add to bar cell array	
	barData(iBar, :) = currBarData(1:bar_grid_eff); % last element belongs to next bar -> remove it
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



function [listing, nFiles] = parseSource(source, feature_type)

% parse source
if iscell(source)  % cell array with path and filenames
    nFiles = length(source);
    listing = cell2struct(source, 'name', nFiles);
else % name of file or folder
    listing = dir(source);
    if length(listing) > 1 % folder
        fileList = dir(fullfile(source, 'beat_activations', ['*.', feature_type]));
        if isempty(fileList)
            fprintf('ERROR extract_bars_from_feature: No feature files in %s found \n', source);
            return;
        end
        nFiles = length(fileList);
        clear listing;
        for iFile=1:nFiles
            fname = strrep(fileList(iFile).name, feature_type, 'wav');
            listing(iFile).name = fullfile(source, fname);
        end
        
    else
        listing(1).name = source;
        nFiles = 1;
    end
end
end
