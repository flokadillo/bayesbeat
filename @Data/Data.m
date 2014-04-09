classdef Data < handle
    % Data Class (represents training and test data)
    properties (SetAccess=private)
        file_list                       % list of files in the dataset
        lab_fln                         % lab file with list of files of dataset
        %         annots_path                     % path to annotations
        bar2file                        % specifies for each bar the file id [nBars x 1]
        bar2cluster                     % specifies for each bar the cluster id [nBars x 1]
        meter                           % meter of each file [nFiles x 1]
        beats                           % beats of each file {nFiles x 1}[n_beats 2]
        n_bars                          % number of bars of each file [nFiles x 1]
        bar_start_id                    % cell [nFiles x 1] [nBeats x 1] with index of first beat of each bar
        full_bar_beats                  % cell [nFiles x 1] [nBeats x 1] 1 = if beat belongs to full bar
        cluster_fln                     % file with cluster id of each bar
        n_clusters                      % total number of clusters
        rhythm_names                    % cell array of rhythmic pattern names
        rhythm2meter                    % specifies for each bar the corresponding meter [R x 2]
        rhythm2meter_state              % specifies for each bar the corresponding meter state [R x 1]
        meter_state2meter               % specifies meter for each meter state [2 x nMeters]
        %         tempo_per_cluster               % tempo of each file ordered by clusters [nFiles x nClusters]
        feats_file_pattern_barPos_dim   % feature values organized by file, pattern, barpos and dim
        pattern_size                    % size of one rhythmical pattern {'beat', 'bar'}
        dataset                         % name of the dataset
    end
    
  
    methods
        
        function obj = Data(lab_fln, train)
            % read lab_fln (a file where all data files are listed)
            if exist(lab_fln, 'file')
                [~, obj.dataset, ext] = fileparts(lab_fln);
                if strcmpi(ext, '.lab')
                    fid = fopen(lab_fln, 'r');
                    obj.file_list = textscan(fid, '%s', 'delimiter', '\n'); obj.file_list = obj.file_list{1};
                    fclose(fid);
                    fln = fullfile('~/diss/src/matlab/beat_tracking/bayes_beat/data', ...
                        [obj.dataset, '-exclude.txt']);
                    if train && exist(fln, 'file')
                        fid = fopen(fln, 'r');
                        exclude_songs = textscan(fid, '%s');
                        fclose(fid);
                        exclude_songs = exclude_songs{1};
                        fprintf('    Excluding %i songs (listed in %s)\n', length(exclude_songs), fln);
                        obj.file_list = obj.file_list(~ismember(obj.file_list, exclude_songs));
                    end
                elseif strcmpi(ext, '.wav') || strcmpi(ext, '.flac')
                    obj.file_list{1} = lab_fln;
                else
                    error('Data: please supply LAB, WAV, or FLAC file instead of %s', lab_fln);
                end
            else
                error('Lab file %s not found\n', lab_fln);
            end
            obj.lab_fln = lab_fln;
        end
        
        function obj = read_pattern_bars(obj, cluster_fln, meters, pattern_size)
            % read cluster_fln (where cluster ids for each bar in the dataset are specified)
            % and generate obj.bar2file, obj.n_bars, obj.meter_state2meter and obj.rhythm2meter
            % TODO: bar2file, obj.n_bars should be loaded not computed!
            if exist(cluster_fln, 'file')
                C = load(cluster_fln);
                obj.bar2file = C.bar2file;
                obj.n_bars = C.file2nBars;
                obj.rhythm_names = C.rhythm_names;
                obj.bar2cluster = C.bar2rhythm;
                obj.rhythm2meter = C.rhythm2meter;
            else
                error('Cluster file %s not found\n', cluster_fln);
            end
            obj.cluster_fln = cluster_fln;
            if ismember(0, obj.bar2cluster), obj.bar2cluster = obj.bar2cluster + 1; end
            obj.n_clusters = max(obj.bar2cluster);
            
            % read pattern_size
            if exist('pattern_size', 'var')
                obj.pattern_size = pattern_size;
            else
                obj.pattern_size = 'bar';
            end
            
%             % read annotations
%             barCounter = 0;
%             for iFile = 1:length(obj.file_list)
%                 [fpath, fname, ~] = fileparts(obj.file_list{iFile});
%                 beats_fln = fullfile(fpath, [fname, '.beats']);
%                 if exist(beats_fln, 'file')
%                     obj.beats{iFile} = load(fullfile(fpath, [fname, '.beats']));
%                 else
%                     error('Beats file %s not found\n', beats_fln);
%                 end
%                 % determine number of bars
%                 if strcmp(obj.pattern_size, 'bar')
%                     [obj.n_bars(iFile), obj.full_bar_beats{iFile}, obj.bar_start_id{iFile}] = obj.get_full_bars(obj.beats{iFile});
%                 else
%                     obj.n_bars(iFile) = size(obj.beats{iFile}, 1) - 1;
%                 end
%                 obj.bar2file(barCounter+1:barCounter + obj.n_bars(iFile)) = iFile;
%                 barCounter = barCounter + obj.n_bars(iFile);
%                 if obj.n_bars(iFile) ~= sum(obj.bar2file == iFile) 
%                     error('%s: Number of bars not consistent !', fname);
%                 end
%             end
%                 
%             % Check consistency cluster_fln - train_lab
%             if length(obj.bar2cluster) ~= length(obj.bar2file)
%                 fprintf('    %s: %i bars\n', cluster_fln, length(obj.bar2cluster));
%                 fprintf('    computed from beat files: %i bars\n', length(obj.bar2file));
%                 error('Number of bars not consistent !');
%             end
%             % find meter of each rhythmic pattern
%             if isempty(obj.meter)
%                 obj = obj.read_meter();
%             end
            obj.meter_state2meter = meters;
            for iR=1:obj.n_clusters
                for iM=1:size(obj.meter_state2meter, 2)
                    if (obj.meter_state2meter(1, iM) == obj.rhythm2meter(iR, 1)) && (obj.meter_state2meter(2, iM) == obj.rhythm2meter(iR, 2))
                        obj.rhythm2meter_state(iR) = iM;
                        break;
                    end
                end
            end
                % find the first bar in data that belongs to cluster iR and
                % look up its meter
%                 m = obj.meter(obj.bar2file(find((obj.bar2cluster == iR), 1)), :);
%                 % TODO: what to do if meter of training data does not match
%                 % meter of system ?
%                 if strcmp(obj.pattern_size, 'bar')
%                     obj.rhythm2meter_state(iR) = find(obj.meter_state2meter(1, :) == m(1));
%                 elseif strcmp(obj.pattern_size, 'beat')
%                     obj.rhythm2meter_state(iR) = 1;
%                 else
%                     error('Meter of training data is not supported by the system')
%                 end
%                 
%             end
        end
        
        function obj = read_meter(obj)
            obj.meter = zeros(length(obj.file_list), 2);
            for iFile = 1:length(obj.file_list)
                [fpath, fname, ~] = fileparts(obj.file_list{iFile});
                meter_fln = fullfile(fpath, [fname, '.meter']);
                if exist(meter_fln, 'file')
                    m = load(meter_fln);
                    if length(m) == 1
                        obj.meter(iFile, 1) = m;
                        obj.meter(iFile, 2) = 4;
                    else
                        obj.meter(iFile, :) = m;
                    end
                else
                    error('Meter file %s not found\n', meter_fln);
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
        
        function tempo_per_cluster = get_tempo_per_cluster(obj)
            tempo_per_cluster = NaN(length(obj.file_list), obj.n_clusters);
            for iFile = 1:length(obj.file_list)
                [fpath, fname, ~] = fileparts(obj.file_list{iFile});
                tempo_fln = fullfile(fpath, [fname, '.bpm']);
                if exist(tempo_fln, 'file')
                    tempo = load(tempo_fln, '-ascii');
                else
                    error('BPM file %s not found\n', tempo_fln);
                end
                % so far, only the first bar of each file is used and assigned the style to the whole file
                styleId = obj.bar2cluster(obj.bar2file == iFile);
                
                % convert to n
                if ~isempty(styleId)
                    tempo_per_cluster(iFile, styleId(1)) = tempo;
                end
            end
            
        end
        
        function obj = extract_feats_per_file_pattern_barPos_dim(obj, whole_note_div, barGrid_eff, ...
                featureDim, featuresFln, featureType, frame_length, reorganize_bars_into_cluster)
            % Extract audio features and sort them according to bars and position
            if exist(featuresFln, 'file') && ~reorganize_bars_into_cluster
                load(featuresFln, 'dataPerFile');
            else
                fprintf('    Extract and organise trainings data: \n');
                for iDim = 1:featureDim
                    fprintf('    dim%i\n', iDim);
                    TrainData = Data.extract_bars_from_feature(obj.file_list, ...
                        featureType{iDim}, whole_note_div, frame_length, obj.pattern_size, 1);
                    temp{iDim} = Data.sort_bars_into_clusters(TrainData.dataPerBar, ...
                        obj.bar2cluster, obj.bar2file);
                end
                [n_files, ~, bar_grid_max] = size(temp{1});
                dataPerFile = cell(n_files, obj.n_clusters, bar_grid_max, featureDim);
                for iDim = 1:featureDim
                    dataPerFile(:, :, :, iDim) = temp{iDim};
                end
                save(featuresFln, 'dataPerFile');
            end
            obj.feats_file_pattern_barPos_dim = dataPerFile;
        end
        
        function belief_func = make_belief_functions(obj, model, file_ids)
            if nargin < 3
                file_ids = 1:length(obj.file_list);
            end
            belief_func = cell(length(file_ids), 2);
            n_states = model.M * model.N * model.R;
            counter = 1;    
%             for i_file = 1:length(obj.file_list)
            for i_file = file_ids(:)'
                [fpath, fname, ~] = fileparts(obj.file_list{i_file});
                beats_fln = fullfile(fpath, [fname, '.beats']);
                if exist(beats_fln, 'file')
                    obj.beats{i_file} = load(fullfile(fpath, [fname, '.beats']));
                else
                    error('Beats file %s not found\n', beats_fln);
                end
                
                rhythm_id = obj.bar2cluster(find(obj.bar2file==i_file, 1));
                t_state = find((obj.meter_state2meter(1, :) == obj.rhythm2meter(rhythm_id, 1)) &...
                    (obj.meter_state2meter(2, :) == obj.rhythm2meter(rhythm_id, 2)));
                r_state = find(model.rhythm2meter_state == t_state);
                M_i = model.Meff(t_state);
                tol_win = floor(0.0875 * model.M / obj.rhythm2meter(rhythm_id, 2)); 
                btype = round(rem(obj.beats{i_file}(:, 2), 1) * 10); % position of beat in a bar: 1, 2, 3, 4
                beats_m = (M_i * (btype-1) / max(btype)) + 1;
                % beat frames
                n_beats_i = size(obj.beats{i_file}, 1);
                i_rows = zeros((tol_win*2+1) * n_beats_i * model.N * length(r_state), 1);
                j_cols = zeros((tol_win*2+1) * n_beats_i * model.N * length(r_state), 1);
                s_vals = ones((tol_win*2+1) * n_beats_i * model.N * length(r_state), 1);
                for iBeat=1:n_beats_i
                    m_support = mod((beats_m(iBeat)-tol_win:beats_m(iBeat)+tol_win) - 1, M_i) + 1;
                    m = repmat(m_support, 1, model.N * length(r_state));
                    n = repmat(1:model.N, length(r_state) * length(m_support), 1);
                    r = repmat(r_state(:), model.N * length(m_support), 1);
                    states = sub2ind([model.M, model.N, model.R], m(:), n(:), r(:));
                    idx = (iBeat-1)*(tol_win*2+1)*model.N*length(r_state)+1:(iBeat)*(tol_win*2+1)*model.N*length(r_state);
                    i_rows(idx) = iBeat;
                    j_cols(idx) = states;
                end
%                 [~, idx, ~] = unique([i_rows, j_cols], 'rows');
                belief_func{counter, 1} = round(obj.beats{i_file}(:, 1) / model.frame_length);
                belief_func{counter, 1}(1) = max([belief_func{counter, 1}(1), 1]);
                belief_func{counter, 2} = logical(sparse(i_rows, j_cols, s_vals, n_beats_i, n_states));
                counter = counter + 1;
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
            %             Indices = crossvalind('Kfold', length(file_list), K);
            C = cvpartition(length(file_list), 'Kfold', K);
            for i=1:K
                fln = fullfile(fpath, [fname, '-fold', num2str(i), ext]);
                dlmwrite(fln, find(test(C, i)), 'delimiter', '\n');
            end
        end
        
        [nBars, beatIdx, barStartIdx] = get_full_bars(beats, tolInt, verbose);
        
        Output = extract_bars_from_feature(source, featExt, barGrid, barGrid_eff, framelength, pattern_size, dooutput);
        
        dataPerFile = sort_bars_into_clusters(dataPerBar, clusterIdx, bar2file);
    end
end
