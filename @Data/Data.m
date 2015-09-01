classdef Data < handle
    % Data Class (represents training and test data)
    properties
        file_list                       % list of files in the dataset
        lab_fln                         % lab file with list of files of dataset
        patt2file                       % specifies for each pattern the file id [nPatts x 1]
        patt2cluster                    % specifies for each pattern the cluster id [nPatts x 1]
        patt2len                        % specifies for each pattern the length [nPatts x 1]
        patt2meter                      % specifies for each pattern the meter [nPatts x 2]
        patt2pattclass                  % specifies for each pattern the pattern class [nPatts x 1]
        pattInfo                        % A structure that stores pattern classes, has n_PattClass elements
        rhythm_names                    % cell array of rhythmic pattern names
        rhythm2meter                    % specifies for each rhythm the corresponding meter [R x 2]
        rhythm2meterID                  % specifies for each rhythm the corresponding meter ID [R x 1]
        rhythm2pattclass                % specifies for each rhythm the corresponding pattern class [R x 1]
        rhythm2section                  % specifies for each rhythm the corresponding section [R x 1]
        rhythm2len_num                  % specifies for each rhythm the corresponding section length numerator [R x 1]
        rhythm2len_den                  % specifies for each rhythm the corresponding section length denominator [R x 1]
        pr                              % rhythm transition probability matrix [R x R]
        prprior                         % rhythm prior probability matrix [1 x R]
        B_pr                            % section transition probability matrix [n_PattClass x n_PattClass]
        B_prprior                       % section prior probability matrix [1 x n_PattClass]    
        meter                           % meter of each file [nFiles x 2]
        beats                           % beats of each file {nFiles x 1}[n_beats 3]
        n_bars                          % number of bars of each file [nFiles x 1]
        bar_start_id                    % cell [nFiles x 1] [nBeats x 1] with index of first beat of each bar
        full_bar_beats                  % cell [nFiles x 1] [nBeats x 1] 1 = if beat belongs to full bar
        cluster_fln                     % file with cluster id of each bar
        n_clusters                      % total number of clusters
        feats_file_pattern_pattPos_dim  % feature values organized by file, pattern, barpos and dim
        feats_silence                   % feature vector of silence
        pattern_size                    % size of one rhythmical pattern {'beat', 'bar'}
        dataset                         % name of the dataset
        pattpos_per_frame               % cell array [nFiles x 1] of bar position (1..bar pos 64th grid) of each frame
        patt_num_per_frame              % cell array [nFiles x 1] of rhythm of each frame
        feat_type                       % cell array (features (extension) to be used) 
    end
    
    
    methods
        function obj = Data(lab_fln, train)
            % read lab_fln (a file where all data files are listed)
            % lab_fln:  text file with filenames and path
            % train:    [0, 1] indicates whether training should be
            %           performed or not
            if exist(lab_fln, 'file')
                [fpath, obj.dataset, ext] = fileparts(lab_fln);
                if strcmpi(ext, '.lab')
                    fid = fopen(lab_fln, 'r');
                    obj.file_list = textscan(fid, '%s', 'delimiter', '\n'); obj.file_list = obj.file_list{1};
                    fclose(fid);
                    % check excluded files
                    fln = fullfile(fpath, [obj.dataset, '-exclude.txt']);
                    if train && exist(fln, 'file')
                        fid = fopen(fln, 'r');
                        exclude_songs = textscan(fid, '%s');
                        fclose(fid);
                        exclude_songs = exclude_songs{1};
                        fprintf('    Excluding %i songs (listed in %s)\n', length(exclude_songs), fln);
                        obj.file_list = obj.file_list(~ismember(obj.file_list, exclude_songs));
                    end
                else
                    obj.file_list{1} = lab_fln;
                end
            else
                error('Lab file %s not found\n', lab_fln);
            end
            % replace paths by absolute paths
            for i_file = 1:length(obj.file_list)
                absolute_path_present = strcmp(obj.file_list{i_file}(1), '~') ...
                    || strcmp(obj.file_list{i_file}(1), filesep) || ...
                    strcmp(obj.file_list{i_file}(2), ':');  % For windows
                if ~absolute_path_present
                    obj.file_list{i_file} = fullfile(pwd, obj.file_list{i_file});
                end
            end
            obj.lab_fln = lab_fln;
        end
        
        function obj = read_pattern_bars(obj, cluster_fln, pattern_size)
            % read cluster_fln (where cluster ids for each bar in the dataset are specified)
            % and generate obj.bar2file, obj.n_bars, and obj.rhythm2meter
            % loads bar2file, n_bars, rhythm_names, bar2cluster, and rhythm2meter
            % creates n_clusters
            if exist(cluster_fln, 'file')
                C = load(cluster_fln);
                obj.patt2file = C.patt2file;
                obj.patt2len = C.patt2len;
                obj.patt2meter = C.patt2meter;
                obj.patt2pattclass = C.patt2pattclass;
                obj.patt2cluster = C.patt2rhythm;
                obj.rhythm_names = C.rhythm_names;
                obj.rhythm2meter = C.rhythm2meter;
                obj.rhythm2meterID = C.rhythm2meterID;
                obj.rhythm2pattclass = C.rhythm2pattclass;
                obj.rhythm2section = C.rhythm2section;
                obj.rhythm2len_num = C.rhythm2len_num;
                obj.rhythm2len_den = C.rhythm2len_den;
                obj.B_pr = C.B_pr;
                obj.B_prprior = C.B_prprior;
                obj.pr = C.pr;
                obj.prprior = C.prprior;
                obj.pattInfo = C.pattInfo; 
                % obj.n_bars = C.file2nBars;
                % obj.rhythm_names = C.rhythm_names;
                % obj.bar2cluster = C.bar2rhythm;
                % obj.rhythm2meter = C.rhythm2meter;
                % obj.rhythm2meterID = C.rhythm2meterID;
                % Adi tala should be 8/4 ideally
                % if any(ismember(obj.rhythm2meter, [8, 4], 'rows'))   % Does not 
                %   fprintf('WARNING Data/read_pattern_bars, 8/4 meter is replaced by 8/8\n');
                %   obj.rhythm2meter = repmat([8,8], size(obj.rhythm2meter, 1), 1);
                % end
                % if isfield(C, 'pr')
                %    % only newer models
                %    obj.pr = C.pr;
                % end
                % if isfield(C, 'prprior')
                %    % only newer models
                %    obj.prprior = C.prprior;
                % end
            else
                error('Cluster file %s not found\n', cluster_fln);
            end
            obj.cluster_fln = cluster_fln;
            if ismember(0, obj.patt2cluster), obj.patt2cluster = ...
                    obj.patt2cluster + 1; end
            obj.n_clusters = max(obj.patt2cluster);
            
            % read pattern_size
            if exist('pattern_size', 'var')
                obj.pattern_size = pattern_size;
            else
                obj.pattern_size = 'bar';
            end
        end
        
        function obj = read_meter(obj)
            obj.meter = zeros(length(obj.file_list), 2);
            meter_files_available = 1;
            for iFile = 1:length(obj.file_list)
                [fpath, fname, ~] = fileparts(obj.file_list{iFile});
                meter_fln = fullfile(strrep(fpath, 'audio', 'annotations/meter'), [fname, '.meter']);
                if exist(meter_fln, 'file')
                    meter = Data.load_annotations_bt(obj.file_list{iFile}, 'meter');
                    if length(meter) == 1
                        obj.meter(iFile, 1) = meter;
                        obj.meter(iFile, 2) = 4;
                        fprintf('Data.read_meter: No denominator in meter file %s found -> adding 4\n', [fname, '.meter']);
                    else
                        obj.meter(iFile, :) = meter;
                    end
                else
                    meter_files_available = 0;
                    fname = meter_fln;
                end
                if ~meter_files_available
                    fprintf('Data.read_meter: One or several meter files not available for this dataset (e.g., %s)\n', fname);
                end
            end
        end
        
        function obj = filter_out_meter(obj, allowed_meters)
            % e.g., allowed_meters = [3, 4]
            if isempty(obj.meter)
                obj = obj.read_meter();
            end
            good_files = ismember(obj.meter, allowed_meters);
            obj.file_list = obj.file_list(good_files);
            obj.meter = obj.meter(good_files);
            if ~isempty(obj.n_bars), obj.n_bars = obj.n_bars(good_files); end
            
            if ~isempty(obj.bar2file)
                bad_files = find(good_files == 0);
                for iFile = 1:length(bad_files)
                    obj.bar2file(obj.bar2file >= bad_files(iFile)) = obj.bar2file(obj.bar2file >= bad_files(iFile)) - 1;
                end
            end
            
            % Check consistency cluster_fln - train_lab
            if sum(obj.n_bars) ~= length(obj.bar2file)
                error('Number of bars not consistent !');
            end
        end
        
        function [tempo_min_per_cluster, tempo_max_per_cluster] = ...
                get_tempo_per_cluster(obj, outlier_percentile)
            if ~exist('outlier_percentile', 'var')
                % discard upper and lower 5% of the beat intervals
                outlier_percentile = 5;
            end
            tempo_min_per_cluster = NaN(length(obj.file_list), obj.n_clusters);
            tempo_max_per_cluster = NaN(length(obj.file_list), obj.n_clusters);
            for iFile = 1:length(obj.file_list)
                [obj.beats{iFile}, error ] = Data.load_annotations_bt(obj.file_list{iFile}, 'beats');
                if error
                    error('Beats file not found\n');
                end
                beat_periods = sort(diff(obj.beats{iFile}(:, 1)), 'descend');
                % ignore outliers
                start_idx = max([floor(length(beat_periods) * ...
                    outlier_percentile / 100), 1]);
                stop_idx = min([floor(length(beat_periods) * (1 - ...
                    outlier_percentile / 100)), length(beat_periods)]);
                if stop_idx >= start_idx
                    beat_periods = beat_periods(start_idx:stop_idx);
                else
                    fprintf('   WARNING @Data/get_tempo_per_cluster.m: outlier_percentile too high!\n');
                    beat_periods = median(beat_periods);
                end
                
                rhythmIDs = unique(obj.patt2cluster(obj.patt2file == iFile));
                tempo_min_per_cluster(iFile, rhythmIDs) = 60/max(beat_periods);
                tempo_max_per_cluster(iFile, rhythmIDs) = 60/min(beat_periods);
                
%                 if ~isempty(rhythmIDs)
%                     maxt = cell(1, obj.n_clusters);
%                     mint = cell(1, obj.n_clusters);
%                     mint(rhythmIDs) = num2cell(ones(1,length(rhythmIDs))*60/max(beat_periods));
%                     maxt(rhythmIDs) = num2cell(ones(1,length(rhythmIDs))*60/min(beat_periods));        
%                     tempo_min_accum(iFile, :) = cellfun(@(x,y) [x y], tempo_min_accum(iFile, :), mint, 'unif', 0);
%                     tempo_max_accum(iFile, :) = cellfun(@(x,y) [x y], tempo_max_accum(iFile, :), maxt, 'unif', 0);
%                 end
            end
%             tempo_min_per_cluster = cellfun(@min, tempo_min_accum);
%             tempo_max_per_cluster = cellfun(@max, tempo_max_accum);
            
            if sum(isnan(max(tempo_max_per_cluster))) > 0 % cluster without tempo
                error('cluster without bar assignment\n');
            end
        end
        
        function obj = extract_feats_per_file_pattern_barPos_dim(obj, whole_note_div, barGrid_eff, ...
                featureDim, featuresFln, featureType, frame_length, reorganize_bars_into_cluster)
            
            obj.feat_type = featureType;
            % Extract audio features and sort them according to bars and position
            if exist(featuresFln, 'file') && ~reorganize_bars_into_cluster
                fprintf('    Loading features from %s\n', featuresFln);
                load(featuresFln, 'dataPerFile');
            else
                fprintf('* Extract and organise training data: \n');
                for iDim = 1:featureDim
                    fprintf('    dim%i\n', iDim);
                    TrainData = Data.extract_patts_from_feature(obj.file_list, ...
                        featureType{iDim}, whole_note_div, frame_length, obj.pattern_size, 1);
                    temp{iDim} = Data.sort_patts_into_clusters(TrainData.dataPerPatt, ...
                        obj.patt2cluster, obj.patt2file);
                end
                [n_files, ~, bar_grid_max] = size(temp{1});
                dataPerFile = cell(n_files, obj.n_clusters, bar_grid_max, featureDim);
                for iDim = 1:featureDim
                    dataPerFile(:, :, :, iDim) = temp{iDim};
                end
                obj.pattpos_per_frame = TrainData.patt_pos_per_frame;
                for i=1:length(TrainData.patt_num_per_frame)
                    obj.patt_num_per_frame{i} = TrainData.patt_num_per_frame{i};
                    obj.patt_num_per_frame{i}(~isnan(obj.patt_num_per_frame{i})) = obj.patt2cluster(obj.patt_num_per_frame{i}(~isnan(obj.patt_num_per_frame{i})));
                end
                pattpos_per_frame = obj.pattpos_per_frame;
                patt_num_per_frame = obj.patt_num_per_frame;
                save(featuresFln, 'dataPerFile', 'pattpos_per_frame', 'patt_num_per_frame', '-v7');
                fprintf('    Saved organized features to %s\n', featuresFln);
            end
            obj.feats_file_pattern_pattPos_dim = dataPerFile;
        end
        
        
        function obj = read_beats(obj)
            % reads beats and downbeats from file and stores them in the
            % data object
            for iFile = 1:length(obj.file_list)
                [obj.beats{iFile}, ~ ] = Data.load_annotations_bt(obj.file_list{iFile}, 'beats');
                if strcmp(obj.pattern_size, 'bar')
                    [obj.n_bars(iFile), obj.full_bar_beats{iFile}, obj.bar_start_id{iFile}] = obj.get_full_bars(obj.beats{iFile});
                else
                    obj.n_bars(iFile) = size(obj.beats{iFile}, 1) - 1;
                end
            end
        end
        
        
        function belief_func = make_belief_functions(obj, model)
            belief_func = cell(length(obj.file_list), 2);
            n_states = model.M * model.N * model.R;
            tol_win_perc_of_ibi = 0.0875;
            %             for i_file = 1:length(obj.file_list)
            for i_file = 1:length(obj.file_list)
                % determine meter state of current file
                r_state = find((obj.rhythm2meter(:, 1) == obj.meter(i_file, 1)) &...
                    (obj.rhythm2meter(:, 2) == obj.meter(i_file, 2)));
                % determine correct M of current file
                M_i = model.Meff(t_state);
                
                tol_win = floor(ol_win_perc_of_ibi * model.M / obj.meter(i_file, 2));
                % determine beat type of each beat
                btype = round(rem(obj.beats{i_file}(:, 2), 1) * 10); % position of beat in a bar: 1, 2, 3, 4
                % compute bar position m for each beat
                beats_m = (M_i * (btype-1) / max(btype)) + 1;
                % beat frames
                n_beats_i = size(obj.beats{i_file}, 1);
                % pre-allocate memory for rows, cols and values
                i_rows = zeros((tol_win*2+1) * n_beats_i * model.N * length(r_state), 1);
                j_cols = zeros((tol_win*2+1) * n_beats_i * model.N * length(r_state), 1);
                s_vals = ones((tol_win*2+1) * n_beats_i * model.N * length(r_state), 1);
                % compute inter beat intervals
                ibi_i = diff(obj.beats{i_file}(:, 1));
                ibi_i = [ibi_i; ibi_i(end)];
                for iBeat=1:n_beats_i
                    % -----------------------------------------------------
                    % Variant 1: tolerance win constant in beats over tempi
                    % -----------------------------------------------------
                    m_support = mod((beats_m(iBeat)-tol_win:beats_m(iBeat)+tol_win) - 1, M_i) + 1;
                    m = repmat(m_support, 1, model.N * length(r_state));
                    n = repmat(1:model.N, length(r_state) * length(m_support), 1);
                    r = repmat(r_state(:), model.N * length(m_support), 1);
                    states = sub2ind([model.M, model.N, model.R], m(:), n(:), r(:));
                    idx = (iBeat-1)*(tol_win*2+1)*model.N*length(r_state)+1:(iBeat)*(tol_win*2+1)*model.N*length(r_state);
                    i_rows(idx) = iBeat;
                    j_cols(idx) = states;
                    % -----------------------------------------------------
                    % Variant 2: Tolerance win constant in time over tempi
                    % -----------------------------------------------------
                    % p=1;
                    % for n_i = model.minN(r_state):model.maxN(r_state)
                    %	tol_win = n_i * tol_win_perc_of_ibi * ibi_i(iBeat) / model.frame_length;
                    %	m_support = mod((beats_m(iBeat)-tol_win:beats_m(iBeat)+tol_win) - 1, M_i) + 1;
                    %	states = sub2ind([model.M, model.N, model.R], m_support(:), ones(size(m_support(:)))*n_i, ones(size(m_support(:)))*r_state);
                    %	j_cols(p:p+length(states)-1) = states;
                    %	i_rows(p:p+length(states)-1) = iBeat;
                    %	p = p + length(states);
                    %   end
                end
                %                 [~, idx, ~] = unique([i_rows, j_cols], 'rows');
                belief_func{i_file, 1} = round(obj.beats{i_file}(:, 1) / model.frame_length);
                belief_func{i_file, 1}(1) = max([belief_func{i_file, 1}(1), 1]);
                belief_func{i_file, 2} = logical(sparse(i_rows, j_cols, s_vals, n_beats_i, n_states));
            end
        end
    end
    
    methods (Static)
        function make_k_folds(lab_fln, K)
            if exist(lab_fln, 'file')
                fid = fopen(lab_fln, 'r');
                file_list = textscan(fid, '%s'); file_list = file_list{1};
                fclose(fid);
            else
                error('Lab file %s not found\n', lab_fln);
            end
            [fpath, fname, ext] = fileparts(lab_fln);
            C = cvpartition(length(file_list), 'Kfold', K);
            for i=1:K
                fln = fullfile(fpath, [fname, '-fold', num2str(i), ext]);
                dlmwrite(fln, find(test(C, i)), 'delimiter', '\n');
            end
        end
        
        % [nBars, beatIdx, barStartIdx] = get_full_bars(beats, tolInt, verbose);
        [nBars, nPatts, beatIdx, barStartIdx, pattStartIdx, secLabelIdx] = get_full_patts(beats, meter, section, tolInt, ...
    verbose)
        
        Output = extract_patts_from_feature(source, featExt, barGrid, barGrid_eff, framelength, pattern_size, dooutput);
        
        dataPerFile = sort_patts_into_clusters(dataPerPatt, clusterIdx, patt2file);
        
        [ data, error ] = load_annotations_bt( filename, ann_type );
    end
end
