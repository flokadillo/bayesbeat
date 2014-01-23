classdef Data
    % Data Class (represents training and test data)
    properties (SetAccess=private)
        file_list                       % list of files in the dataset
        lab_fln                         % lab file with list of files of dataset
        %         annots_path                     % path to annotations
        bar2file                        % specifies for each bar the file id [nBars x 1]
        bar2cluster                     % specifies for each bar the cluster id [nBars x 1]
        meter                           % meter of each file [nFiles x 1]
        n_bars                          % number of bars of each file [nFiles x 1]
        cluster_fln                     % file with cluster id of each bar
        n_clusters                      % total number of clusters
        rhythm_names                    % cell array of rhythmic pattern names
        rhythm2meter                    % specifies for each bar the corresponding meter state
        meter_state2meter               % specifies meter for each meter state
        %         tempo_per_cluster               % tempo of each file ordered by clusters [nFiles x nClusters]
        feats_file_pattern_barPos_dim   % feature values organized by file, pattern, barpos and dim
        pattern_size                    % size of one rhythmical pattern {'beat', 'bar'}
    end
    
    methods(Static)
        
    end
    
    methods
        
        function obj = Data(lab_fln)
            % read lab_fln (a file where all data files are listed)
            if exist(lab_fln, 'file')
                [~, dataset, ext] = fileparts(lab_fln);
                if strcmpi(ext, '.lab')
                    fid = fopen(lab_fln, 'r');
                    obj.file_list = textscan(fid, '%s', 'delimiter', '\n'); obj.file_list = obj.file_list{1};
                    fclose(fid);
                    fln = fullfile('~/diss/src/matlab/beat_tracking/bayes_beat/data', ...
                        [dataset, '-train_ids.txt']);
                    if exist(fln, 'file')
                        ok_songs = load(fln);
                        fprintf('    Excluding %i songs\n', length(obj.file_list) - length(ok_songs));
                        obj.file_list = obj.file_list(ok_songs);
                    end
                elseif strcmpi(ext, '.wav')
                    obj.file_list{1} = lab_fln;
                end
            else
                error('Lab file %s not found\n', lab_fln);
            end
            obj.lab_fln = lab_fln;
        end
               
        function obj = read_pattern_bars(obj, cluster_fln, meters, pattern_size)
            % read cluster_fln (where cluster ids for each bar in the dataset are specified)
            % and generate obj.bar2file, obj.n_bars, obj.meter_state2meter and obj.rhythm2meter
            if exist(cluster_fln, 'file')
                obj.bar2cluster = load(cluster_fln, '-ascii');
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
            
            % read pattern_labels
            fln = strrep(cluster_fln, '.txt', '-rhythm_labels.txt');
            if exist(fln, 'file')
                fid = fopen(fln, 'r');
                obj.rhythm_names = textscan(fid, '%s', 'delimiter', '\n'); obj.rhythm_names = obj.rhythm_names{1};
                fclose(fid);
            else
                % just call the rhythms by their ids
                obj.rhythm_names = cellfun(@(x) num2str(x), num2cell(1:obj.n_clusters)', 'UniformOutput',false);
            end
            % read annotations
            obj.n_bars = zeros(length(obj.file_list), 1);
            obj.bar2file = zeros(length(obj.bar2cluster), 1);
            barCounter = 0;
            for iFile = 1:length(obj.file_list)
                [fpath, fname, ~] = fileparts(obj.file_list{iFile});
                beats_fln = fullfile(fpath, [fname, '.beats']);
                if exist(beats_fln, 'file')
                    beats = load(fullfile(fpath, [fname, '.beats']));
                else
                    error('Beats file %s not found\n', beats_fln);
                end
                % determine number of bars
                if strcmp(obj.pattern_size, 'bar')
                    [obj.n_bars(iFile), ~, ~] = obj.get_full_bars(beats);
                else
                    obj.n_bars(iFile) = size(beats, 1) - 1;
                end
                obj.bar2file(barCounter+1:barCounter + obj.n_bars(iFile)) = iFile;
                barCounter = barCounter + obj.n_bars(iFile);
            end
            % Check consistency cluster_fln - train_lab
            if sum(obj.n_bars) ~= length(obj.bar2file)
                error('Number of bars not consistent !');
            end
            % find meter of each rhythmic pattern
            if isempty(obj.meter)
                obj = obj.read_meter();
            end
            obj.meter_state2meter = meters;
            for iR=1:obj.n_clusters
                % find the first bar in data that belongs to cluster iR and
                % look up its meter
                m = obj.meter(obj.bar2file(find((obj.bar2cluster == iR), 1)), :);
                % TODO: what to do if meter of training data does not match
                % meter of system ?
                if strcmp(obj.pattern_size, 'bar')
                    obj.rhythm2meter(iR) = find(obj.meter_state2meter(1, :) == m(1));
                elseif strcmp(obj.pattern_size, 'beat')
                    obj.rhythm2meter(iR) = 1;
                else
                    error('Meter of training data is not supported by the system')
                end
                    
            end
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
