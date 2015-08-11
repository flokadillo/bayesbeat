function [feat_from_bar_and_gmm] = organise_feats_into_bars(obj, whole_note_div)
% obj = organise_feats_into_bars(obj, whole_note_div)
% ----------------------------------------------------------------------
%INPUT parameter:
% whole_note_div     : number of gridpoints of one whole note [64]
%
%OUTPUT parameter:
% feat_from_bar_and_gmm : [nBars, nGMMs, featDim] cell array with features
%                           vectors
% obj.bar2file :     [nBars x 1]
% 03.08.2015 by Florian Krebs
% ----------------------------------------------------------------------
% Parse parameters
if nargin == 1,
    whole_note_div = 64;
end
fprintf('* Organize feature values into bars ...\n');
% load meter and beats for all files
% determine maximum bar grid (corresponding to the longest bar)
if strcmp(obj.pattern_size, 'bar')
    bar_grid_max = ceil(max(whole_note_div * obj.meters(:, 1) ./ ...
        obj.meters(:, 2)));
else
    % beat length patterns, assume quarter note beats
    bar_grid_max = round(whole_note_div / 4);
end
nchar = 0;
feat_dim = length(obj.feature.feat_type);
feat_from_bar_and_gmm = [];
bar2file = zeros(sum(obj.n_bars), 1);
idLastBar = 0;
%main loop over all files
for iFile=1:length(obj.file_list)
    [~, fname, ~] = fileparts(obj.file_list{iFile});
    fprintf(repmat('\b', 1, nchar));
    nchar = fprintf('      %i/%i) %s', iFile, length(obj.file_list), fname);
    if isempty(obj.beats{iFile})
        if dooutput, fprintf('Error loading annotations, skipping %s\n', ...
                fname); end
        continue;
    end
    % For one bar to be extracted, the downbeat of the bar itself and the
    % next downbeat has to be present in the annotations. Otherwise, it is
    % discarded
    b1 = find(obj.beats{iFile}(:, 2) == 1);
    if (length(b1) <= 1) && strcmp(obj.pattern_size, 'bar')
        if dooutput,
            fprintf(repmat('\b', 1, nchar));
            nchar = fprintf('    %s contains only one bar -> skip it\n', ...
                fname);
        end
    else
        meter = obj.meters(iFile, :);
        beats = obj.beats{iFile};
        % Effective number of bar positions (depending on meter)
        if strcmp(obj.pattern_size, 'bar')
            bar_grid_eff = ceil(whole_note_div * meter(:, 1) ./ ...
                obj.meters(iFile, 2));
        else
            % beat-length patterns: suppose quarter beats
            bar_grid_eff = whole_note_div / 4;
        end
        % Collect feature values and determine the corresponding position
        % in a bar
        % load feature values from file and up/downsample to frame_length
        E = obj.feature.load_feature(obj.file_list{iFile});
        fr = 1/obj.feature.frame_length;
        % if feature vector is shorter than annotations, copy last value
        if length(E) * obj.feature.frame_length < beats(end, 1)
            E = [E; E(end, :)];
        end
        if beats(end, 1) > length(E) * obj.feature.frame_length;
            fprintf('     Warning: beat annotations longer than audio file\n');
            nchar = 0;
            beats = beats(beats(:, 1) <= length(E) * ...
                obj.feature.frame_length, :);
        end
        if strcmp(obj.pattern_size, 'bar')
            nBars = obj.n_bars(iFile);
            barStartIdx = obj.bar_start_id{iFile};
        else
            nBars = size(beats, 1) - 1;
            barStartIdx = 1:nBars;
        end
        beatsBarPos = (0:meter(1)) * whole_note_div / obj.meters(iFile, 2) ...
            + 1;
        barData = cell(nBars, bar_grid_eff, feat_dim);
        bar_pos_per_frame = nan(length(E), 'single');
        pattern_per_frame = nan(length(E), 'single');
        for iBar=1:nBars
            % compute start and end frame of bar using fr
            first_frame_of_bar = floor(beats(barStartIdx(iBar), 1) * fr) + 1; % first frame of bar
            first_frame_of_next_bar = floor(beats(barStartIdx(iBar)+meter(1), 1) * fr) + 1; % first frame of next bar          
            % extract feature for this bar
            featBar = E(first_frame_of_bar:first_frame_of_next_bar, :);
            % set up time frames of bar, subtract half frame (1/(2*fr)) to yield center of frame
            t = (first_frame_of_bar:first_frame_of_next_bar) / fr - 1/(2*fr);
            % interpolate to find bar position of each audio frame
            barPosLin = round(interp1(beats(barStartIdx(iBar):barStartIdx(iBar)+meter(1), 1), beatsBarPos, t,'linear','extrap'));
            barPosLin(barPosLin < 1) = 1;
            % bar position 64th grid per frame
            bar_pos_per_frame(first_frame_of_bar:first_frame_of_next_bar-1) = barPosLin(1:end-1);
            pattern_per_frame(first_frame_of_bar:first_frame_of_next_bar-1) = ones(first_frame_of_next_bar-first_frame_of_bar, 1) * iBar;
            % group all feature values that belong to the same barPos
            labels = [repmat(barPosLin(:), feat_dim, 1) ...             %# Replicate the row indices
                kron(1:feat_dim, ones(1, numel(barPosLin))).'];  %'# Create column indices
            currBarData = accumarray(labels, featBar(:), ...
                [bar_grid_eff+2, 2], @(x) {x});
            % add to bar cell array; last element belongs to next bar -> remove it
            barData(iBar, :, :) = currBarData(1:bar_grid_eff, :); 
        end
        % interpolate missing values
        if sum(sum(cellfun(@isempty, barData))) > 0
            data_means = cellfun(@mean, barData);
            [i_bars, cols, ~] = size(barData);
            times = 1:cols;
            for i = 1:i_bars
                % we assume that if a feature value is missing in one
                % feature dimension it will also be missing in all the
                % others
                missing_data_position = isnan(data_means(i, :, 1));
                if sum(missing_data_position) > 0
                    barData(i, missing_data_position, :) = ...
                        num2cell(interp1(times(~missing_data_position), ...
                        squeeze(data_means(i, ~missing_data_position, :)), ...
                        times(missing_data_position), 'linear', 'extrap'));
                end
            end
        end
        if ~isempty(barData)
            [nNewBars, currBarGrid, ~] = size(barData);
            % for shorter metrical cycles fill in empty cells
            if currBarGrid < bar_grid_max,
                barData = [barData, cell(size(barData, 1), bar_grid_max-size(barData, 2), 2)];
            end
            feat_from_bar_and_gmm = [feat_from_bar_and_gmm; barData];
            bar2file((idLastBar + 1):(idLastBar + nNewBars)) = iFile;
%             pattern_per_frame{iFile} = pattern_per_frame{iFile} + idLastBar;
            idLastBar = idLastBar + nNewBars;
        end
    end
end
% remove output
fprintf(repmat('\b', 1, nchar));
% store results
obj.bar2file = bar2file(:);
end % FUNCTION end