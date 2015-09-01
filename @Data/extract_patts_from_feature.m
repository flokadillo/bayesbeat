function [Output] = extract_patts_from_feature(source, feature_type, whole_note_div, frame_length, pattern_size, dooutput)
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
% Output.dataPerPatt : [nBars x whole_note_div] matrix of double 
%                       from this, cellfun(@mean, Output.dataPerBar)
%                       computes the mean of each bar and position and can
%                       be plotted by plot(mean(cellfun(@mean, Output.dataPerBar)))
% Output.bar2file   : [1 x nBars] vector
% Output.fileNames  : [nFiles x 1] vector
% Output.file2meter : [nFiles x 2]
% 
%
% 26.7.2012 by Florian Krebs
% ----------------------------------------------------------------------

% ####################### PARAMETERS ######################################
% set parameters
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
Output.dataPerPatt = []; idLastPatt = 0;
Output.fileNames = cell(nFiles, 1);
Output.file2meter = zeros(nFiles, 2);
beats_all = cell(nFiles, 1);
fprintf('* Organize feature values (%s) into bars/sections ...\n', feature_type);

% load meter, sections, and beats for all files
for iFile=1:nFiles
    [meter, error1] = Data.load_annotations_bt(listing(iFile).name, 'meter');
    [section, error2] = Data.load_annotations_bt(listing(iFile).name, 'section');
    section = section(:)';
    if strcmp(pattern_size, 'bar')     % Set it explicitly here!!
        section = 1;
    end
    [beats_all{iFile}, error3] = Data.load_annotations_bt(listing(iFile).name, 'beats');
    if error1
        Output.file2meter(iFile, 1) = max(beats_all{iFile}(:, 2));
        Output.file2meter(iFile, 2) = 4;
        fprintf('    WARNING: Denominator of the meter unknown, assuming quarter note beats\n');
    else
        Output.file2meter(iFile, :) = meter;
    end
    if error2
        Output.file2section{iFile} = 1;
        Output.maxSecLen(iFile) = meter(1);
        fprintf('    WARNING: Section information not be read, assuming bar length sections\n');
    else
        Output.file2section{iFile} = section;
        Output.sec_len{iFile} = diff([section section(1) + meter(1)]);
        max_sec_len(iFile) = max(Output.sec_len{iFile});
    end
end
% determine maximum bar grid (corresponding to the longest bar)
if strcmp(pattern_size, 'bar')
    bar_grid_max = ceil(max(whole_note_div * Output.file2meter(:, 1) ./ Output.file2meter(:, 2)));
elseif strcmp(pattern_size, 'section')
    % section length patterns
    bar_grid_max = ceil(max(whole_note_div * max_sec_len(:) ./ Output.file2meter(:, 2)));
else
    % beat length patterns, assume quarter note beats
    bar_grid_max = round(whole_note_div / 4);    
end
nchar = 0;
%main loop over all files
for iFile=1:nFiles
    [~, fname, ~] = fileparts(listing(iFile).name);
    fprintf(repmat('\b', 1, nchar));
    nchar = fprintf('      %i/%i) %s', iFile, nFiles, fname);
    if isempty(beats_all{iFile})
        if dooutput, fprintf('Error loading annotations, skipping %s\n', fname); end
        continue;
    end
    Output.fileNames{iFile} = listing(iFile).name;
    wav_fln = listing(iFile).name;
    % for one bar to be extracted, the downbeat of the bar itself and the
    % next downbeat has to be present in the annotations. Otherwise, it is
    % discarded
    b1 = find(beats_all{iFile}(:, 2) == 1);
    if (length(b1) <= 1) && strcmp(pattern_size, 'bar')
        if dooutput, 
            fprintf(repmat('\b', 1, nchar));
            fprintf('    %s contains only one bar -> skip it\n', fname); 
            nchar = 0;
        end
    else
        % effective number of bar positions (depending on meter)
        if strcmp(pattern_size, 'bar')
            bar_grid_eff = ceil(whole_note_div * Output.file2meter(iFile, 1) ./ Output.file2meter(iFile, 2));
        elseif strcmp(pattern_size, 'section')
            bar_grid_eff = ceil(whole_note_div * Output.sec_len{iFile} ./ Output.file2meter(iFile, 2));
        else
            % beat-length patterns: suppose quarter beats 
            bar_grid_eff = whole_note_div / 4;
        end
        % collect feature values and determine the corresponding position
        % in a bar
        
        
        [pattData, nchar, Output.patt_pos_per_frame{iFile}, Output.patt_num_per_frame{iFile},...
            Output.sec_label_per_frame{iFile}, Output.bar_num_per_frame{iFile}, Output.sec_label_per_pattern{iFile}] = ...
            get_feature_at_bar_grid(wav_fln, feature_type, beats_all{iFile}, whole_note_div, bar_grid_eff, ...
            frame_length, pattern_size, Output.file2meter(iFile, :), ...
            Output.file2section{iFile}, Output.sec_len{iFile}, nchar);
        if ~isempty(pattData)
            [nNewPatts, currBarGrid] = size(pattData);
            % for triple meter fill in empty cells
            if currBarGrid < bar_grid_max,
                pattData = [pattData, cell(size(pattData, 1), bar_grid_max-size(pattData, 2))];
            elseif currBarGrid > whole_note_div,
                bar_grid_max = currBarGrid;
            end
            Output.dataPerPatt = [Output.dataPerPatt; pattData];          
            Output.patt2file((idLastPatt + 1):(idLastPatt + nNewPatts)) = iFile;
            Output.patt2sec((idLastPatt + 1):(idLastPatt + nNewPatts)) = Output.sec_label_per_pattern{iFile}(:);
            Output.patt_num_per_frame{iFile} = Output.patt_num_per_frame{iFile} + idLastPatt;
            Output.patt_len((idLastPatt + 1):(idLastPatt + nNewPatts)) = Output.sec_len{iFile}(Output.sec_label_per_pattern{iFile}(:));
            idLastPatt = idLastPatt + nNewPatts;
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


function [pattData, nchar, patt_pos_per_frame, patt_num_per_frame, ...
    sec_label_per_frame, bar_num_per_frame, secLabelIdx] = get_feature_at_bar_grid(wav_fln, ...
    feature_type, beats, whole_note_div, bar_grid_eff, frame_length, ...
    pattern_size, meter, section, secLens, nchar)
% pattData   [nPatts x whole_note_div] cell array features values per bar and bargrid

% load feature values from file and up/downsample to frame_length
[E, fr] = load_features(wav_fln, feature_type, frame_length);
% if feature vector to short for annotations, copy last value
if length(E)/fr < beats(end, 1), E = [E; E(end)]; end

if beats(end, 1) > length(E) * frame_length;
    fprintf('     Warning: beat annotations longer than audio file\n');
    nchar = 0;
    beats = beats(beats(:, 1) <= length(E) * frame_length, :);
end

if strcmp(pattern_size, 'bar') || strcmp(pattern_size, 'section')
    % [nBars, ~, barStartIdx] = Data.get_full_patts(beats, section, pattern_size);  
    [~, nPatts, ~, barStartIdx, pattStartIdx, secLabelIdx] = Data.get_full_patts(beats, meter, section);
else
    nPatts = size(beats, 1) - 1;
    pattStartIdx = 1:nPatts+1;
    barStartIdx = pattStartIdx;
    secLabelIdx = ones(size(pattStartIdx));
    nBars = nPatts;
    meter = [1; 4];
end

for k = 1:length(pattStartIdx)
    barNum(k) = sum(beats(pattStartIdx(k),1) >=  beats(barStartIdx,1)); 
end

pattLen = secLens(secLabelIdx);

pattData = cell(nPatts, max(bar_grid_eff));
% pattData(:) = {nan};
patt_pos_per_frame = nan(size(E), 'single');
patt_num_per_frame = nan(size(E), 'single');
sec_label_per_frame = nan(size(E), 'single');
bar_num_per_frame = nan(size(E), 'single');

for iBar=1:nPatts
	% beatsBarPos = ((0:meter(1)) * whole_note_div / meter(2)) + 1;
    beatsBarPos = ((0:pattLen(iBar)) * whole_note_div / meter(2)) + 1;
    
    % compute start and end frame of bar using fr
    first_frame_of_patt = floor(beats(pattStartIdx(iBar), 1) * fr) + 1; % first frame of bar
	first_frame_of_next_patt = floor(beats(pattStartIdx(iBar+1), 1) * fr) + 1; % first frame of next bar

	% extract feature for this pattern
    featBar = E(first_frame_of_patt:first_frame_of_next_patt);

	% set up time frames of bar
    t = (first_frame_of_patt:first_frame_of_next_patt) / fr - 1/(2*fr);
     
	% interpolate to find bar position of each audio frame
	pattPosLin = round(interp1(beats(pattStartIdx(iBar):pattStartIdx(iBar+1), 1), beatsBarPos, t,'linear','extrap'));
    pattPosLin(pattPosLin < min(beatsBarPos)) = min(beatsBarPos);
    pattPosLin(pattPosLin > max(beatsBarPos)) = max(beatsBarPos);
    % pattern position 64th grid per frame
    patt_pos_per_frame(first_frame_of_patt:first_frame_of_next_patt-1) = pattPosLin(1:end-1);
    patt_num_per_frame(first_frame_of_patt:first_frame_of_next_patt-1) = ones(first_frame_of_next_patt-first_frame_of_patt, 1) * iBar;
    bar_num_per_frame(first_frame_of_patt:first_frame_of_next_patt-1) = barNum(iBar); 
    sec_label_per_frame(first_frame_of_patt:first_frame_of_next_patt-1) = ones(first_frame_of_next_patt-first_frame_of_patt, 1) * secLabelIdx(iBar);
	% group all feature values that belong to the same barPos
% 	currBarData = accumarray(barPosLin', featBar(:), [], @(x) {x});
    currBarData = accumarray(pattPosLin', featBar(:), [bar_grid_eff(secLabelIdx(iBar))+1, 1], @(x) {x});       % Here was another bug, changed it
	% add to bar cell array	
	pattData(iBar, 1:bar_grid_eff(secLabelIdx(iBar))) = ...
        currBarData(1:bar_grid_eff(secLabelIdx(iBar))); % last element belongs to next bar -> remove it
end
secLabelIdx(end) = [];
% % interpolate missing values
% if sum(sum(cellfun(@isempty, pattData))) > 0
%     data_means = cellfun(@mean, pattData);
%     [rows, cols] = size(pattData);
%     times = 1:cols;
%     for i = 1:rows
%         missing_data = isnan(data_means(i, :));
%         if sum(missing_data) > 0
%             pattData(i, missing_data) = num2cell(interp1(times(~missing_data), ...
%                 data_means(i, ~missing_data), times(missing_data), 'linear', 'extrap'));
%         end
%     end
% end


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
