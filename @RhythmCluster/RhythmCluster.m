classdef RhythmCluster < handle
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
        train_file_list     % list of training files
%         beat_ann_list       % list of beat annotation files
        data_save_path      % path where cluster ids per bar are stored
        exclude_songs_fln   % vector of file ids that contain more than one bar and have supported meter
        n_clusters          % number of clusters
        pattern_size        % size of one rhythmical pattern {'beat', 'bar'}
        data_per_bar        % [nBars x feat_dim*bar_grid]
        data_per_song       % [nSongs x feat_dim*bar_grid]
        bar2file            % [1 x nBars] vector
        bar_2_cluster
        file_2_cluster
        bar_2_meter         % [nBars x 2]
        file_2_meter        % [nFiles x 2]
        rhythm2meter        % [nRhythms x 2]
        cluster_transition_matrix
        cluster_transitions_fln
    end
    
    methods
        function obj = RhythmCluster(dataset, feat_type, frame_length, data_save_path, pattern_size)
            %  obj = RhythmCluster(dataset, feat_type, frame_length, data_save_path, pattern_size)
            %  Construct Rhythmcluster object
            % ----------------------------------------------------------------------
            %INPUT parameter:
            % dataset                 : path to lab file (list of training files)
            %
            %OUTPUT parameter:
            % obj
            %
            % 09.04.2013 by Florian Krebs
            % ----------------------------------------------------------------------
            if nargin == 1
                feat_type = {'lo230_superflux.mvavg.normZ', 'hi250_superflux.mvavg.normZ'};
                frame_length = 0.02;
                data_save_path = './data';
                pattern_size = 'bar';
            end
            obj.feature = Feature(feat_type, frame_length);
            obj.clusters_fln = '/tmp/cluster_assignments.txt';
            
            
            obj.data_save_path = data_save_path;
            if exist('pattern_size', 'var')
                obj.pattern_size = pattern_size;
            else
                obj.pattern_size = 'bar';
            end
            % load list of training files
            if strfind(dataset, '.lab')
                obj.train_lab_fln = dataset;
                [~, obj.dataset, ~] = fileparts(dataset); 
            else
                obj.train_lab_fln = fullfile('~/diss/data/beats/lab_files', [dataset, '.lab']);
                obj.dataset = dataset;
            end
                
            if exist(obj.train_lab_fln, 'file')
                fid = fopen(obj.train_lab_fln, 'r');
                obj.train_file_list = textscan(fid, '%s', 'delimiter', '\n');
                obj.train_file_list = obj.train_file_list{1};
                fclose(fid);
                % beat annotation path is usually under
                % annotations/beats/xxx.beats
%                 obj.beat_ann_list = cellfun(@(x) strrep(x, 'audio', 'annotations/beats'), obj.train_file_list, 'UniformOutput', 0);
%                 obj.beat_ann_list = cellfun(@(x) strrep(x, '.wav', '.beats'), obj.train_file_list, 'UniformOutput', 0);
%                 obj.beat_ann_list = cellfun(@(x) strrep(x, '.flac', '.beats'), obj.train_file_list, 'UniformOutput', 0);
            else
               error('ERROR RhythmCluster.m: %s not found\n', obj.train_lab_fln);
            end
        end
        
        function make_feats_per_song(obj, whole_note_div)
            
            obj.make_feats_per_bar(whole_note_div);
            
            % compute mean per song
            obj.data_per_song = NaN(length(obj.train_file_list), size(obj.data_per_bar, 2));
            for iCol=1:size(obj.data_per_bar, 2)
                obj.data_per_song(:, iCol) = accumarray(obj.bar2file', obj.data_per_bar(:, iCol), [], @mean);
            end
            % find songs that contain more than one bar and have
            % allowed meter
            exclude_song_ids = ~ismember(1:length(obj.train_file_list), unique(obj.bar2file));
            %             ok_songs = ismember(1:length(obj.train_file_list), unique(obj.bar2file));
            
            % save features (mean of all bars of one song)
            obj.feat_matrix_fln = fullfile(obj.data_save_path, ['onsetFeat-', ...
                num2str(obj.feature.feat_dim), 'd-', obj.dataset, '-songs.txt']);
            dlmwrite(obj.feat_matrix_fln, obj.data_per_song(~exclude_song_ids, :), 'delimiter', '\t', 'precision', 4);
            fprintf('Saved data per song to %s\n', obj.feat_matrix_fln);
            exclude_song_ids = find(exclude_song_ids);
            if ~isempty(exclude_song_ids)
                obj.exclude_songs_fln = fullfile(obj.data_save_path, [obj.dataset, '-exclude.txt']);
                fid = fopen(obj.exclude_songs_fln, 'w');
                for i=1:length(exclude_song_ids)
                    fprintf(fid, '%s\n', obj.train_file_list{exclude_song_ids(i)});
                end
                fclose(fid);
                fprintf('Saved files to be excluded (%i) to %s\n', length(exclude_song_ids), obj.exclude_songs_fln);
%                 dlmwrite(obj.exclude_songs_fln, find(exclude_song_ids));
            end
            
            %             obj.ok_songs_fln = fullfile(obj.data_save_path, [obj.dataset, '-train_ids.txt']);
            %             dlmwrite(obj.ok_songs_fln, unique(obj.bar2file)');
            
        end
        
        function make_feats_per_bar(obj, whole_note_div)
            if exist(obj.train_lab_fln, 'file')
                fprintf('    Found %i files in %s\n', length(obj.train_file_list), obj.dataset);
                dataPerBar = [];
                for iDim =1:obj.feature.feat_dim
                    Output = Data.extract_bars_from_feature(obj.train_file_list, obj.feature.feat_type{iDim}, ...
                        whole_note_div, obj.feature.frame_length,obj.pattern_size, 1);
                    dataPerBar = [dataPerBar, cellfun(@mean, Output.dataPerBar)];
                end
                obj.bar2file = Output.bar2file;
                obj.data_per_bar = dataPerBar;
                obj.file_2_meter = Output.file2meter;
                obj.bar_2_meter = Output.file2meter(obj.bar2file, :);
                % save bars to file
                obj.feat_matrix_fln = fullfile(obj.data_save_path, ['onsetFeat_', ...
                    num2str(obj.feature.feat_dim), 'd_', obj.dataset, '.txt']);
                dlmwrite(obj.feat_matrix_fln, dataPerBar, 'delimiter', '\t', 'precision', 4);
                fprintf('    Saved data per bar to %s\n', obj.feat_matrix_fln);
            else
                error('    %s not found', obj.train_lab_fln)
            end
        end
        
        function load_feats_per_bar(obj)
            obj.feat_matrix_fln = fullfile(obj.data_save_path, ['onsetFeat_', ...
                num2str(obj.feature.feat_dim), 'd_', obj.dataset, '.txt']);
            obj.data_per_bar = dlmread(obj.feat_matrix_fln, '\t');
        end
        
        function load_feats_per_song(obj)
            obj.feat_matrix_fln = fullfile(obj.data_save_path, ['onsetFeat-', ...
                num2str(obj.feature.feat_dim), 'd-', obj.dataset, '-songs.txt']);
            obj.data_per_song = dlmread(obj.feat_matrix_fln, '\t');
        end
        
        function ca_fln = do_clustering(obj, n_clusters, type, bad_meters)
            % type = 'bars' : cluster data_per_bar
            % type = 'songs': cluster data_per_song
            % bad_meters    : [1 x nMeters] numerator of meters to ignore
            %                   (e.g. [3, 8, 9])
            %             system(['python ~/diss/projects/rhythm_patterns/do_clustering.py -k ', ...
            %                 num2str(n_clusters), ' ', obj.feat_matrix_fln]);
            [file2nBars, ~] = hist(obj.bar2file, 1:length(obj.train_file_list));
            fprintf('    WARNING: so far only 2 different meters supported!\n');
            if strcmpi(type, 'bars')
                S = obj.data_per_bar;
                meter_per_item = obj.bar_2_meter;
            else
                S = obj.data_per_song;
                meter_per_item = obj.file_2_meter;
            end
            meters = unique(meter_per_item, 'rows');
            if size(meters, 1) == 1 % only one meter
                n_items_per_meter = size(meter_per_item, 1);
            else % count items per meter
                n_items_per_meter = hist(meter_per_item(:, 1), sort(meters(:, 1), 'ascend'));
            end
            n_clusters_per_meter = ceil(n_items_per_meter * n_clusters / sum(n_items_per_meter));
            if sum(n_clusters_per_meter) > n_clusters % check because of ceil
               [~, idx] = max(n_clusters_per_meter); % remove one cluster from meter that has most clusters
               n_clusters_per_meter(idx) = n_clusters_per_meter(idx) - sum(n_clusters_per_meter) + n_clusters;
            end
            
            % normalise data
            bar_pos = size(S, 2) / obj.feature.feat_dim;
            for i_dim = 1:obj.feature.feat_dim
                vals = S(:, (i_dim-1)*bar_pos+1:bar_pos*i_dim);
                vals = vals - nanmean(vals(:));
                vals = vals / nanstd(vals(:));
                S(:, (i_dim-1)*bar_pos+1:bar_pos*i_dim) = vals;
            end
            opts = statset('MaxIter', 200);
            % cluster different meter separately
            ctrs = zeros(n_clusters, size(S, 2));
            cidx = zeros(size(S, 1), 1);
            p = 1; 
            S(isnan(S)) = -99;
            obj.rhythm2meter = zeros(n_clusters, 2);
            for iMeter=1:size(meters, 1)
                idx_i = (meter_per_item(:, 1) == meters(iMeter, 1)) & (meter_per_item(:, 2) == meters(iMeter, 2));
                if n_clusters_per_meter(iMeter) > 1 % more than one cluster per meter -> kmeans
                    [cidx(idx_i), ctrs(p:p+n_clusters_per_meter(iMeter)-1, :)] = kmeans(S(idx_i, :), n_clusters_per_meter(iMeter), 'Distance', 'sqEuclidean', 'Replicates', 5, 'Options', opts);
                    cidx(idx_i) = cidx(idx_i) + p - 1;
                    obj.rhythm2meter(p:p+n_clusters_per_meter(iMeter)-1, :) = repmat(meters(iMeter, :), n_clusters_per_meter(iMeter), 1);
                    p=p+n_clusters_per_meter(iMeter);
                else % only one item per meter -> no kmeans necessary
                    ctrs(p, :) = mean(S(idx_i, :));
                    cidx(idx_i) = p;
                    obj.rhythm2meter(p, :) = meters(iMeter, :);
                    p=p+1;
                end
            end
%             S(isnan(S)) = -999;
            
%             [cidx, ctrs] = kmeans(S, n_clusters, 'Distance', 'sqEuclidean', 'Replicates', 5, 'Options', opts);
            ctrs(ctrs==-99) = nan;
            
            plot_cols = ceil(sqrt(n_clusters));
            h = figure( 'Visible','off');
            set(h, 'Position', [100 100 n_clusters*100 n_clusters*100]);
            items_per_cluster = hist(cidx, n_clusters);
            col = hsv(obj.feature.feat_dim);
            for c = 1:n_clusters
                subplot(ceil(n_clusters/plot_cols), plot_cols, c)
                hold on
                for fdim = 1:obj.feature.feat_dim
                    data = ctrs(c, (fdim-1)*bar_pos+1:fdim*bar_pos);
                    data = data - min(data);
                    data = data / max(data);
                    data = data + fdim;
                    stairs([data, data(end)], 'Color', col(fdim, :));
                end
                xlabel('bar position')
                title(sprintf('cluster %i (%i %s)', c, items_per_cluster(c), type));
                xlim([1 length(data)])
            end
            outfile = ['/tmp/patterns-', obj.dataset, '-kmeans-', type, '-', num2str(n_clusters), '.png'];
            fprintf('    Writing patterns to %s\n', outfile);
            % save to png
            print(h, outfile, '-dpng');
            
            % save cluster assignments
            if strcmpi(type, 'bars')
                obj.clusters_fln = fullfile(obj.data_save_path, ['ca-', obj.dataset, '-', ...
                    num2str(obj.feature.feat_dim), 'd-', num2str(n_clusters), 'R-kmeans.mat']);
            else
                % read index of  valid songs
                if exist(obj.exclude_songs_fln, 'f')
                    exclude_songs = load(obj.exclude_songs_fln, '-ascii');
                else
                    exclude_songs = [];
                end
                meter = zeros(length(obj.train_file_list), 1);
                fileCounter = 0;
                bar2rhythm = [];
                for iFile = 1:length(obj.train_file_list)
                    if ismember(iFile, exclude_songs)
                        continue;
                    end
                    [beats, ~ ] = Data.load_annotations_bt(obj.train_file_list{iFile}, 'beats');
                    % filter out meters that are not 3 or 4
                    meter(fileCounter+1) = max(beats(:, 3));
                    if exist('bad_meters', 'var')
                        if ismember(meter(fileCounter+1), bad_meters)
                            fprintf('    Warning: Skipping %s because of meter (%i)\n', obj.train_file_list{iFile}, meter(fileCounter+1));
                            continue
                        end
                    end
                    fileCounter = fileCounter + 1;
                    % determine number of bars
%                     [nBars(iFile), ~] = Data.get_full_bars(beats);
                    if file2nBars(iFile) > 0
                        patternId = cidx(fileCounter);
                        bar2rhythm = [bar2rhythm; ones(file2nBars(iFile), 1) * patternId];
                    end
                    
                end
                cidx = bar2rhythm;
                obj.clusters_fln = fullfile(obj.data_save_path, ['ca-', obj.dataset, '-', ...
                        num2str(obj.feature.feat_dim), 'd-', num2str(n_clusters),'R-kmeans-songs.mat']);
            end
            
%             fprintf('writing %i bar-cluster assignments to %s\n', length(obj.bar_2_cluster), obj.clusters_fln);
            rhythm2meter = obj.rhythm2meter;
            obj.bar_2_cluster = cidx;
            bar2rhythm = cidx;
            bar2file = obj.bar2file;
            for i = 1:n_clusters
                rhythm_names{i} = [obj.dataset, '-kmeans_', num2str(i)];
            end
            save(obj.clusters_fln, '-v7.3', 'bar2rhythm', 'bar2file', ...
                'file2nBars', 'rhythm_names', 'rhythm2meter');
            fprintf('    Saved bar2rhythm, bar2file, file2nBars, rhythm_names, rhythm2meter to %s\n', obj.clusters_fln);
            obj.n_clusters = n_clusters;
            ca_fln = obj.clusters_fln;
        end
        
        function compute_cluster_transitions(obj)
            A = zeros(obj.n_clusters, obj.n_clusters);
            for iFile=1:length(obj.train_file_list)
                bars = find(obj.bar2file==iFile);
                for iBar=bars(1:end-1)
                    A(obj.bar_2_cluster(iBar), obj.bar_2_cluster(iBar+1)) = A(obj.bar_2_cluster(iBar), obj.bar_2_cluster(iBar+1)) + 1;
                end
            end
            A = bsxfun(@rdivide, A , sum(A , 2));
            obj.cluster_transition_matrix = A;
            obj.cluster_transitions_fln = fullfile(obj.data_save_path, ['cluster_transitions-', obj.dataset, '-', ...
                num2str(obj.feature.feat_dim), 'd-', num2str(obj.n_clusters), '.txt']);
            dlmwrite(obj.cluster_transitions_fln, A);
            fprintf('Saved transition matrix to %s\n', obj.cluster_transitions_fln);
        end
        
        
        function [ca_fln] = make_cluster_assignment_file(obj, clusterType, rhythm_names)
            % [bar2rhythm] = make_cluster_assignment_file(trainLab)
            %   Creates vector bar2rhythm that assigns each
            %   bar in trainLab to the pattern specified by the dancestyle annotation.
            % ----------------------------------------------------------------------
            %INPUT parameter:
            % trainLab          : filename of labfile (e.g., 'boeck.lab' or 'ballroom.lab').
            %                       a labfile is a textfile with paths to files that are analyzed
            % clusterType         : {'meter', 'dancestyle', 'rhythm', 'none'}
            %                   'meter': bars are clustered according to the meter (functions reads .meter file);
            %                   'dancestyle', according to the genre (functions reads .dancestyle file))
            %                   'none' : put all bars into one single
            %                   cluster
            % clustAssFln       : if clusterType='auto', specify a textfile, with a
            %                       pattern id for each file (in the same order as in trainLab)
            % rhythm_names      : cell array of strings
            %
            %OUTPUT parameter:
            % bar2rhythm       : [nBars x 1] assigns each bar to a pattern
            %
            % 09.07.2013 by Florian Krebs
            % ----------------------------------------------------------------------
%             addpath('~/diss/src/matlab/libs/matlab_utils');
            if nargin == 1
                clusterType = 'auto';
            end
            if strcmpi(clusterType, 'auto')
                songClusterIds = load(obj.clusters_fln, '-ascii');
            end
            
            bar2rhythm = [];
            obj.bar2file = [];
%             dancestyles = {'ChaCha', 'Jive', 'Quickstep', 'Rumba', 'Samba', 'Tango', 'VienneseWaltz', 'Waltz'};
            
            % read list of training files
            fid = fopen(obj.train_lab_fln, 'r');
            temp = textscan(fid, '%s', 'delimiter', '\n');
            fileNames = temp{1};
            fclose(fid);
            if isempty(obj.exclude_songs_fln)
                ok_songs = 1:length(fileNames);
            else
                fid = fopen(obj.exclude_songs_fln, 'r');
                exclude_songs = textscan(fid, '%s');
                fclose(fid);
                exclude_songs = exclude_songs{1};
                ok_songs = find(~ismember(obj.train_file_list, exclude_songs));
            end
            meter = zeros(length(ok_songs), 1);
            fileCounter = 0;
            nBars = zeros(length(ok_songs), 1);
            obj.file_2_cluster = zeros(length(ok_songs), 1);
            for iFile = 1:length(ok_songs)
%                 [annotsPath, fname, ~] = fileparts(fileNames{ok_songs(iFile)});
                [beats, error] = Data.load_annotations_bt( obj.train_file_list{ok_songs(iFile)}, 'beats');
                if error, error('no beat file found\n'); end
                if size(beats, 2) < 2
                    error('Downbeat annotations missing for %s\n', obj.train_file_list{ok_songs(iFile)}); 
                end
                meter(fileCounter+1) = max(beats(:, 3));
                
                % get pattern id of file
                switch lower(clusterType)
                    case 'meter'
                        patternId = meter(fileCounter+1);
                    case 'dancestyle'
                        style = Data.load_annotations_bt( obj.train_file_list{ok_songs(iFile)}, 'dancestyle');
                        patternId = find(strcmp(rhythm_names, style));
                        if isempty(patternId)
                           fprintf('Please add %s to the rhythm_names\n', style);
                        end
                        obj.file_2_cluster(iFile) = patternId;
                    case 'auto'
                        patternId = songClusterIds(iFile);
                    case 'rhythm'
                        style = Data.load_annotations_bt( strrep(obj.train_file_list{ok_songs(iFile)}, 'audio', 'annotations/rhythm'), 'rhythm');
                        patternId = find(strcmp(rhythm_names, style));
                    case 'none'
                        patternId = 1;
                end
                fileCounter = fileCounter + 1;
                % determine number of bars
                if strcmp(obj.pattern_size, 'beat')
                    nBars(iFile) = size(beats, 1) - 1;
                else
                    [nBars(iFile), ~] = Data.get_full_bars(beats);
                end
                bar2rhythm = [bar2rhythm; ones(nBars(iFile), 1) * patternId];
                obj.bar2file = [obj.bar2file; ones(nBars(iFile), 1) * ok_songs(iFile)];
            end
     
            if strcmp(clusterType, 'meter')
                % conflate patternIds
                meters = unique(bar2rhythm);
                temp = 1:length(meters);
                temp2(meters) = temp;
                bar2rhythm = temp2(bar2rhythm)';               
            end
            obj.n_clusters = max(bar2rhythm);
            obj.rhythm2meter = zeros(obj.n_clusters, 2);
            for iR=1:obj.n_clusters
                % find a bar/file that represents rhythm iR
                file_id = obj.bar2file(find(bar2rhythm==iR, 1));
%                 [annotsPath, fname, ~] = fileparts(fileNames{file_id});
                meter = Data.load_annotations_bt( strrep(obj.train_file_list{file_id}, 'audio', 'annotations/meter'), 'meter');
                % TODO: what to do if meter of training data does not match
                % meter of system ?
                if strcmp(obj.pattern_size, 'bar')
                    obj.rhythm2meter(iR, :) = meter;
                elseif strcmp(obj.pattern_size, 'beat')
                    obj.rhythm2meter(iR, :) = [1, 4];
                else
                    error('Meter of training data is not supported by the system')
                end
            end
            obj.bar_2_cluster = bar2rhythm;
            ca_fln = fullfile(obj.data_save_path, ['ca-', obj.dataset, '-', ...
                num2str(obj.feature.feat_dim), 'd-', num2str(obj.n_clusters),'R-', clusterType, '.mat']);
            bar2rhythm = bar2rhythm;
            file2nBars = nBars;
            rhythm2meter = obj.rhythm2meter;
            if ~exist('rhythm_names', 'var')
                for i = unique(bar2rhythm(:))'
                    rhythm_names{i} = [clusterType, num2str(i)];
                end
            end
            bar2file = obj.bar2file;
            save(ca_fln, '-v7.3', 'bar2rhythm', 'bar2file', 'file2nBars', 'rhythm_names', 'rhythm2meter');
            fprintf('writing %s\n', ca_fln);
        end
    end
end

