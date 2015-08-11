classdef Data < handle
% This class represents the data to be used for the bayes_beat system.

    properties
        file_list            % list of files in the dataset
        lab_fln              % lab file with list of files of dataset
        dataset              % name of the dataset
        bar2file             % specifies for each bar the file id [nBars x 1]
        meters               % meter of each file [nFiles x 2]
        beats                % beats of each file {nFiles x 1}[n_beats 2]
        n_bars               % number of (complete) bars of each file [nFiles x 1]
        bar_start_id         % cell [nFiles x 1] [nBeats x 1] with index of first beat of each bar
        full_bar_beats       % cell [nFiles x 1] [nBeats x 1] 1 = if beat belongs to full bar
        cluster_fln          % file with cluster id of each bar
        features_organised   % feature values organized by file, pattern, barpos and dim
                             % cell(n_files, n_clusters, bar_grid_max, featureDim)
        feats_silence        % feature vector of silence
        pattern_size         % size of one rhythmical pattern {'beat', 'bar'}
        feature              % Feature object
        clustering           % a RhythmCluster object
    end
    
    
    methods
        function obj = Data(lab_fln, feat_type, frame_length, pattern_size)
            % obj = Data(lab_fln, feat_type, frame_length, pattern_size)
            %   Creates Data object.
            % ----------------------------------------------------------------------
            % INPUT Parameter:
            %   lab_fln             : filename of textfile that holds all
            %                           the filenames of the dataset or
            %                           cell array that holds the
            %                           filenames. File has to have
            %                           extension .lab
            %   feat_type           : cell array with one cell per feature
            %                           dimension [e.g., feat_type =
            %                           {'lo230_superflux.mvavg', ...
            %                           'hi250_superflux.mvavg'}]
            %   frame_length        : audio frame length
            %   pattern_size        : 'bar' or 'beat'
            %
            % 10.08.2015 by Florian Krebs
            % ----------------------------------------------------------------------
            if iscell(lab_fln) % cell array of filenames
                obj.file_list = lab_fln;
            elseif exist(lab_fln, 'file') % textfile with filenames
                [~, obj.dataset, ext] = fileparts(lab_fln);
                if strcmp(ext, '.lab')
                    fid = fopen(lab_fln, 'r');
                    obj.file_list = textscan(fid, '%s', 'delimiter', '\n');
                    obj.file_list = obj.file_list{1};
                    fclose(fid);
                    obj.lab_fln = lab_fln;
                else % audio file itself given
                    obj.file_list{1} = lab_fln;
                end
            else
                error('Lab file %s not found\n', lab_fln);
            end
            % replace paths by absolute paths
            for i_file = 1:length(obj.file_list)
                absolute_path_present = strcmp(obj.file_list{i_file}(1), ...
                    '~') || strcmp(obj.file_list{i_file}(1), filesep) || ...
                    strcmp(obj.file_list{i_file}(2), ':');  % For windows
                if ~absolute_path_present
                    obj.file_list{i_file} = fullfile(pwd, ...
                        obj.file_list{i_file});
                end
            end
            % create feature object
            obj.feature = Feature(feat_type, frame_length);
            obj.pattern_size = pattern_size;
        end
        
        function [] = cluster_from_features(obj, n_clusters, input_args)
            %  [] = cluster_from_features(obj, n_clusters, input_args)
            %  Performs clustering of the bars by a kmeans in the feature
            %  space.
            % ------------------------------------------------------------
            %INPUT parameter:
            % n_clusters    number of clusters (should be at least one per 
            %               time signature)
            % input_args    can be either a cell array with features already
            %               organised by bar and bar position or the number 
            %               of grid points per whole note
            %
            % 10.08.2015 by Florian Krebs
            % ------------------------------------------------------------
            obj.clustering = RhythmCluster(obj);
            % get features per bar
            if iscell(input_args)
                feat_from_bar_and_gmm = input_args;
            else
                feat_from_bar_and_gmm = ...
                    obj.organise_feats_into_bars(input_args);
            end
            obj.clustering.cluster_from_features(feat_from_bar_and_gmm, ...
                n_clusters);
        end
        
        function [] = cluster_from_labels(obj, cluster_type)
            %  [] = cluster_from_labels(obj, cluster_type)
            %  Performs clustering of the bars by labels.
            % ------------------------------------------------------------
            %INPUT parameter:
            % cluster_type  {'meter', 'rhythm', 'none'}
            %
            % 10.08.2015 by Florian Krebs
            % ------------------------------------------------------------
            obj.clustering = RhythmCluster(obj);
            obj.clustering.cluster_from_labels(cluster_type);
        end
        
        
        function [] = read_pattern_bars(obj, cluster_fln)
            % Reads bar2file, n_bars, rhythm_names, bar2cluster, and
            % rhythm2meter from file.
            if exist(cluster_fln, 'file')
                obj.clustering = RhythmCluster(obj);
                C = load(cluster_fln);
                obj.bar2file = C.bar2file(:);
                obj.n_bars = C.file2nBars;
                obj.clustering.rhythm_names = C.rhythm_names;
                obj.clustering.bar2cluster = C.bar2rhythm;
                obj.clustering.rhythm2meter = C.rhythm2meter;
                if isfield(C, 'pr')
                    % only newer models
                    obj.clustering.pr = C.pr;
                end
            else
                error('Cluster file %s not found\n', cluster_fln);
            end
            obj.clustering.cluster_fln = cluster_fln;
            obj.clustering.n_clusters = max(obj.clustering.bar2cluster);
        end
        
        function meters = get.meters(obj)
            % This method calls obj.load_annotations_bt and loads the time
            % signature for each file to a matrix [nFiles x 2]
            if isempty(obj.meters)
                obj.meters = zeros(length(obj.file_list), 2);
                meter_files_available = 1;
                for iFile = 1:length(obj.file_list)
                    if strcmp(obj.pattern_size, 'bar')
                        [fpath, fname, ~] = fileparts(obj.file_list{iFile});
                        meter_fln = fullfile(strrep(fpath, 'audio', ...
                            'annotations/meter'), [fname, '.meter']);
                        if exist(meter_fln, 'file')
                            meter = Data.load_annotations_bt(...
                                obj.file_list{iFile}, 'meter');
                            if length(meter) == 1
                                obj.meters(iFile, 1) = meter;
                                obj.meters(iFile, 2) = 4;
                                fprintf(['Data.read_meter: No denominator', ...
                                    'in meter file %s found -> adding 4\n'], ...
                                    [fname, '.meter']);
                            else
                                obj.meters(iFile, :) = meter;
                            end
                        else
                            meter_files_available = 0;
                            fname = meter_fln;
                        end
                        if ~meter_files_available
                            fprintf(['Data.read_meter: One or several ', ...
                                'meter files not available for this', ...
                                'dataset (e.g., %s)\n'], fname);
                        end
                    else
                        obj.meters(iFile, :) = [1, 4];
                    end
                end
            end
            meters = obj.meters;
        end
        
        function beats = get.beats(obj)
            % This method calls obj.load_annotations_bt and loads beats and
            % downbeats for each file to a cell array {nFiles x 1}[n_beats 2]
            if isempty(obj.beats) % only do this if not already done
                for iFile = 1:length(obj.file_list)
                    [obj.beats{iFile}, ~ ] = Data.load_annotations_bt(...
                        obj.file_list{iFile}, 'beats');
                    if strcmp(obj.pattern_size, 'bar')
                        [obj.n_bars(iFile), obj.full_bar_beats{iFile}, ...
                            obj.bar_start_id{iFile}] = ...
                            obj.get_full_bars(obj.beats{iFile});
                    else
                        obj.n_bars(iFile) = size(obj.beats{iFile}, 1) - 1;
                    end
                end
            end
            beats = obj.beats;
        end
        
        function [tempo_min_per_cluster, tempo_max_per_cluster] = ...
                get_tempo_per_cluster(obj, outlier_percentile)
            % Computes a min and max tempo value for each song in the
            % dataset. You can set an outlier_percentile to remove outlier
            % from the set of beat intervals per song.
            if ~exist('outlier_percentile', 'var')
                % discard upper and lower 5% of the beat intervals
                outlier_percentile = 0;
            end
            if isempty(obj.clustering.n_clusters),
                error('Please perform clustering first\n');
            end
            tempo_min_per_cluster = NaN(length(obj.file_list), ...
                obj.clustering.n_clusters);
            tempo_max_per_cluster = NaN(length(obj.file_list), ...
                obj.clustering.n_clusters);
            beats = obj.beats;
            for iFile = 1:length(obj.file_list)
                beat_periods = sort(diff(beats{iFile}(:, 1)), 'descend');
                % ignore outliers
                start_idx = max([floor(length(beat_periods) * ...
                    outlier_percentile / 100), 1]);
                stop_idx = min([floor(length(beat_periods) * (1 - ...
                    outlier_percentile / 100)), length(beat_periods)]);
                if stop_idx >= start_idx
                    beat_periods = beat_periods(start_idx:stop_idx);
                else
                    fprintf(['   WARNING @Data/get_tempo_per_cluster.m:', ...
                        'outlier_percentile too high!\n']);
                    beat_periods = median(beat_periods);
                end
                styleId = unique(obj.clustering.bar2cluster(...
                    obj.bar2file == iFile));
                if ~isempty(styleId)
                    tempo_min_per_cluster(iFile, styleId) = ...
                        60/max(beat_periods);
                    tempo_max_per_cluster(iFile, styleId) = ...
                        60/min(beat_periods);
                end
            end
            if sum(isnan(max(tempo_max_per_cluster))) > 0 % cluster without tempo
                error('cluster without bar assignment\n');
            end
        end
        
        feat_from_bar_and_gmm = organise_feats_into_bars(obj, ...
            whole_note_div);
        % obj = organise_feats_into_bars(obj, whole_note_div)
        % Load features and organise them into bars and bar positions using
        % <whole_note_div> cells per whole note in a bar.
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
        
        [] = sort_bars_into_clusters(obj, dataPerBar);
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
    end
    
    methods (Static)
        
        [nBars, beatIdx, barStartIdx] = get_full_bars(beats, tolInt, ...
            verbose);
        %  [nBars, beatIdx, barStartIdx] = get_full_bars(beats, tolInt, verbose)
        %  returns complete bars within a sequence of beats. If there are multiple
        %  time signature within the beat sequence, only the main meter is
        %  counted.
        % ----------------------------------------------------------------------
        %INPUT parameter:
        % beats                     : [nBeats x 2]
        %                               first col: beat times, second col metrical position
        % tolInt                    : pauses are detected if the interval between
        %                               two beats is bigger than tolInt times the last beat period
        %                               [default 1.8]
        %
        %OUTPUT parameter:
        % nBars                     : number of complete bars
        % beatIdx                   : [nBeats x 1] of boolean: determines if beat
        %                               belongs to a full bar (=1) or not (=0)
        % barStartIdx               : index of first beat of each bar
        %
        % 11.07.2013 by Florian Krebs
        % ----------------------------------------------------------------------
        
        [data, error] = load_annotations_bt(filename, ann_type);
        % [data, error] = load_annotations_bt( filename, ann_type)
        %
        %   Loads annotations according to the extension of the file from file.
        %   Supports .beats, .bpm, .meter, .rhythm, .dancestyle, .onsets.
        %   The following cases are supported
        %   1) The extension does not match the supported ones. Please specify
        %   ann_type in this case
        %   2) The path represents the audio-folder (dataset/audio). Here, the
        %   path is updated to point to dataset/annotations/xxx
        % ----------------------------------------------------------------------
        % INPUT Parameter:
        % filename          : filename of annotation file including extension
        % datatype          : [optional, only if extension is not standard]
        %                       e.g., 'beats', 'bpm', 'onsets', 'meter'
        %
        % OUTPUT Parameter:
        % data              : annotations
        % error             : > 0 if some of the requested entities could not be
        %                       loaded
        %
        % 16.05.2014 by Florian Krebs
        % ----------------------------------------------------------------------
    end
end
