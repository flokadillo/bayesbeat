classdef RhythmCluster
    % WORKFLOW:
    %
    % if patterns are given via labels
    % 1) RhythmCluster
    % 2) make_cluster_assignment_file
    
    properties
        feature             % feature object
        feat_matrix_fln     % fln where feature values of dataset are stored
        clusters_fln        % fln where cluster ids are stored
        dataset             % training dataset on which clustering is performed
        train_lab_fln       % lab file with training data files
        data_save_path      % path where cluster ids per bar are stored
        ok_songs_fln        % vector of file ids that contain more than one bar and have supported meter
        n_clusters          % number of clusters
    end
    
    methods
        function obj = RhythmCluster(feat_type, frame_length, dataset, data_save_path)
            obj.feature = Feature(feat_type, frame_length);
            obj.clusters_fln = '/tmp/cluster_assignments.txt';
            obj.dataset = dataset;
            obj.train_lab_fln = fullfile('~/diss/data/beats/', [dataset, '.lab']);
            obj.data_save_path = data_save_path;
        end
        
        function obj = make_feats_per_song(obj, dataset, data_save_path)
            if ~exist('data_save_path', 'var')
                obj.data_save_path = '~/diss/src/matlab/beat_tracking/bayes_beat/data/';
            else
                obj.data_save_path = data_save_path;
            end
            obj.dataset = dataset;
            obj.train_lab_fln = ['~/diss/data/beats/', dataset, '.lab'];
            if exist(obj.train_lab_fln, 'file')
                fid = fopen(obj.train_lab_fln, 'r');
                fileList = textscan(fid, '%s');
                fileList = fileList{1};
                fclose(fid);
                fprintf('Found %i files in %s\n', length(fileList), dataset);
                dataPerBar = [];
                for iDim =1:obj.feature.feat_dim
                    Output = Data.extract_bars_from_feature(fileList, obj.feature.feat_type{iDim});
                    dataPerBar = [dataPerBar, cellfun(@mean, Output.dataPerBar)];
                end
                % compute mean per song
                dataPerSong = NaN(length(fileList), size(dataPerBar, 2));
                for iCol=1:size(dataPerBar, 2)
                    dataPerSong(:, iCol) = accumarray(Output.bar2file', dataPerBar(:, iCol), [], @mean);
                end
                % find songs that contain more than one bar and have
                % allowed meter
                ok_songs = ismember(1:length(fileList), unique(Output.bar2file));
                obj.feat_matrix_fln = fullfile(obj.data_save_path, ['onsetFeat-', ...
                    num2str(obj.feature.feat_dim), 'd-', dataset, '-songs.txt']);
                dlmwrite(obj.feat_matrix_fln, dataPerSong(ok_songs, :), 'delimiter', '\t', 'precision', 4);
                obj.ok_songs_fln = fullfile(obj.data_save_path, [dataset, '-train_ids.txt']);
                dlmwrite(obj.ok_songs_fln, unique(Output.bar2file)');
            else
                error('%s not found', obj.train_lab_fln)
            end
            
        end
        
        function obj = make_feats_per_bar(obj, dataset, data_save_path)
            if ~exist('data_save_path', 'var')
                obj.data_save_path = '~/diss/src/matlab/beat_tracking/bayes_beat/data/';
            else
                obj.data_save_path = data_save_path;
            end
            obj.dataset = dataset;
            obj.train_lab_fln = ['~/diss/data/beats/', dataset, '.lab'];
            if exist(obj.train_lab_fln, 'file')
                fid = fopen(obj.train_lab_fln, 'r');
                fileList = textscan(fid, '%s');
                fileList = fileList{1};
                fclose(fid);
                fprintf('Found %i files in %s\n', length(fileList), dataset);
                dataPerBar = [];
                for iDim =1:obj.feature.feat_dim
                    Output = Data.extract_bars_from_feature(fileList, obj.feature.feat_type{iDim});
                    dataPerBar = [dataPerBar, cellfun(@mean, Output.dataPerBar)];
                end
                % save bars to file
                obj.feat_matrix_fln = fullfile(obj.data_save_path, ['onsetFeat_', ...
                    num2str(obj.feature.feat_dim), 'd_', dataset, '.txt']);
                dlmwrite(obj.feat_matrix_fln, dataPerBar, 'delimiter', '\t', 'precision', 4);
            else
                error('%s not found', obj.train_lab_fln)
            end
        end
        
        function obj = do_clustering(obj, n_clusters)
            system(['python ~/diss/projects/rhythm_patterns/do_clustering.py -k ', ...
                num2str(n_clusters), ' ', obj.feat_matrix_fln]);
            %             X = load(obj.feat_matrix_fln);
            %             idx = kmeans(X, n_clusters);
            obj.clusters_fln = '/tmp/cluster_assignments.txt';
            obj.n_clusters = n_clusters;
        end
        
        function [] = copy_song_patterns_to_bars(obj)
            % if whole songs are clustered, the cluster id is copied once
            % for each bar to have the same format that is used when bars
            % are clustered separately
            songClusterIds= load(obj.clusters_fln, '-ascii');
            bar2pattern = [];
            % read list of training files
            fid = fopen(obj.train_lab_fln, 'r');
            temp = textscan(fid, '%s');
            fileNames = temp{1};
            fclose(fid);
            % read index of  valid songs
            ok_songs = load(obj.ok_songs_fln, '-ascii');
            
            meter = zeros(length(fileNames), 1);
            fileCounter = 0;
            nBars = zeros(length(fileNames), 1);
            for iFile = 1:length(fileNames)
                if ~ismember(iFile, ok_songs)
                    continue;
                end
                beats = load(regexprep(fileNames{iFile}, '.wav.*', '.beats'));
                countTimes = round(rem(beats(:, 2), 1) * 10);
                % filter out meters that are not 3 or 4
                meter(fileCounter+1) = max(countTimes);
                if ~ismember(meter(fileCounter+1), [3, 4])
                    continue
                end
                fileCounter = fileCounter + 1;
                % determine number of bars
                [nBars(iFile), ~] = Data.get_full_bars(beats);
                if nBars(iFile) > 0
                    patternId = songClusterIds(fileCounter);
                    bar2pattern = [bar2pattern; ones(nBars(iFile), 1) * patternId];
                end
            end
            dlmwrite(fullfile(obj.data_save_path, ['ca-', obj.dataset, '-', ...
                num2str(obj.feature.feat_dim), 'd-', num2str(max(songClusterIds)+1),'-songs.txt']), bar2pattern);
        end
        
        function [] = make_cluster_assignment_file(obj, clusterType, rhythm_names)
            % [bar2pattern] = make_cluster_assignment_file(trainLab)
            %   Creates vector bar2pattern that assigns each
            %   bar in trainLab to the pattern specified by the dancestyle annotation.
            % ----------------------------------------------------------------------
            %INPUT parameter:
            % trainLab          : filename of labfile (e.g., 'boeck.lab' or 'ballroom.lab').
            %                       a labfile is a textfile with paths to files that are analyzed
            % clusterType         : {'meter', 'dancestyle', 'auto'} if clusterType='meter', bars are
            %                       clustered according to the meter (functions reads .meter file);
            %                       if clusterType='dancestyle', according to the genre (functions reads .dancestyle file))
            % clustAssFln       : if clusterType='auto', specify a textfile, with a
            %                       pattern id for each file (in the same order as in trainLab)
            %
            %OUTPUT parameter:
            % bar2pattern       : [nBars x 1] assigns each bar to a pattern
            %
            % 09.07.2013 by Florian Krebs
            % ----------------------------------------------------------------------
            addpath('~/diss/src/matlab/libs/matlab_utils');
            if nargin == 1
                clusterType = 'auto';
            end
            if strcmpi(clusterType, 'auto')
                songClusterIds = load(obj.clusters_fln, '-ascii');
            end
            
            
            bar2pattern = [];
            dancestyles = {'ChaCha', 'Jive', 'Quickstep', 'Rumba', 'Samba', 'Tango', 'VienneseWaltz', 'Waltz'};
            
            % read list of training files
            fid = fopen(obj.train_lab_fln, 'r');
            temp = textscan(fid, '%s');
            fileNames = temp{1};
            fclose(fid);
            if ~isempty(obj.ok_songs_fln)
                ok_songs = load(obj.ok_songs_fln, '-ascii');
            else
                ok_songs = 1:length(fileNames);
            end
            meter = zeros(length(ok_songs), 1);
            fileCounter = 0;
            nBars = zeros(length(ok_songs), 1);
            % for each file in the cluster
            for iFile = 1:length(ok_songs)
                beats = load(regexprep(fileNames{ok_songs(iFile)}, '.wav.*', '.beats'));
                countTimes = round(rem(beats(:, 2), 1) * 10);
                % filter out meters that are not 3 or 4
                meter(fileCounter+1) = max(countTimes);
                %                 if ~ismember(meter(fileCounter+1), [3, 4])
                %                     continue
                %                 end
                % get pattern id of file
                [annotsPath, fname, ~] = fileparts(fileNames{iFile});
                fprintf('- %s\n', fname);
                switch lower(clusterType)
                    case 'meter'
                        patternId = meter(fileCounter+1) - 2;
                    case 'dancestyle'
                        [ data, ~ ] = loadAnnotations( annotsPath, [fname,'.wav'], 's', 1 );
                        patternId = find(strcmp(dancestyles, data.style));
                    case 'auto'
                        patternId = songClusterIds(iFile);
                    case 'rhythm'
                        fln = strrep(fileNames{iFile}, '.wav', '.rhythm');
                        if exist(fln, 'file')
                            fid = fopen(fln, 'r');
                            style = textscan(fid, '%s');
                            fclose(fid);
                        end
                        patternId = find(strcmp(rhythm_names, style{1}{1}));
                end
                fileCounter = fileCounter + 1;
                % determine number of bars
                [nBars(iFile), ~] = Data.get_full_bars(beats);
                bar2pattern = [bar2pattern; ones(nBars(iFile), 1) * patternId];
            end
            obj.n_clusters = max(bar2pattern);
            dlmwrite(fullfile(obj.data_save_path, ['ca-', obj.dataset, '-', ...
                num2str(obj.feature.feat_dim), 'd-', num2str(obj.n_clusters),'.txt']), bar2pattern);
            % write rhythm names to file
            fln = fullfile(obj.data_save_path, ['ca-', obj.dataset, '-', ...
                num2str(obj.feature.feat_dim), 'd-', num2str(obj.n_clusters),'-rhythm_labels.txt']);
            fid = fopen(fln, 'w');
            for i=1:length(rhythm_names)
                fprintf(fid, '%s\n', rhythm_names{i});
            end
            fclose(fid);
            save(fln, 'rhythm_names', 'ascii');
        end
    end
end

