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
        rhythm_names        % {R x 1} strings
        file_2_nBars        % [nFiles x 1] number of bars per file
        cluster_transition_matrix
        cluster_transitions_fln
    end
    
    methods
        function obj = RhythmCluster(dataset, feat_type, frame_length, ...
                data_save_path, pattern_size)
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
                feat_type = {'lo230_superflux.mvavg', 'hi250_superflux.mvavg'};
                frame_length = 0.02;
                data_save_path = './data';
                pattern_size = 'bar';
            end
            obj.feature = Feature(feat_type, frame_length);
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
                train_data = Data(obj.train_lab_fln, 1);
                obj.train_file_list = train_data.file_list;
            else
                error('ERROR RhythmCluster.m: %s not found\n', obj.train_lab_fln);
            end
        end
        
        
        function make_feats_per_song(obj, whole_note_div)
            % make_feats_per_song: Computes mean of features for all bar
            % position within a song
            %
            % INPUT:    whole_note_div : number of bar positions for a whole
            %           note
            % OUTPUT:   obj.data_per_song [nSongs x feat_dim*bar_grid]
            % ------------------------------------------------------------
            
            % extract features per bar position and create obj.bar2file,
            % obj.data_per_bar, obj.file_2_meter, and obj.bar_2_meter
            obj.make_feats_per_bar(whole_note_div);
            % compute mean per song
            obj.data_per_song = NaN(length(obj.train_file_list), ...
                size(obj.data_per_bar, 2));
            for iCol=1:size(obj.data_per_bar, 2) % loop over (feat_dim*bar_grid)
                obj.data_per_song(:, iCol) = accumarray(obj.bar2file', ...
                    obj.data_per_bar(:, iCol), [], @mean);
            end
            % find songs that contain more than one bar and have
            % allowed meter
            exclude_song_ids = ~ismember(1:length(obj.train_file_list), ...
                unique(obj.bar2file));
            % save features (mean of all bars of one song)
            obj.feat_matrix_fln = fullfile(obj.data_save_path, ['onsetFeat-', ...
                num2str(obj.feature.feat_dim), 'd-', obj.dataset, '-songs.txt']);
            dlmwrite(obj.feat_matrix_fln, obj.data_per_song(~exclude_song_ids, :), ...
                'delimiter', '\t', 'precision', 4);
            fprintf('Saved data per song to %s\n', obj.feat_matrix_fln);
            % Save list of excluded files to file
            exclude_song_ids = find(exclude_song_ids);
            if ~isempty(exclude_song_ids)
                obj.exclude_songs_fln = fullfile(obj.data_save_path, [obj.dataset, ...
                    '-exclude.txt']);
                fid = fopen(obj.exclude_songs_fln, 'w');
                for i=1:length(exclude_song_ids)
                    fprintf(fid, '%s\n', obj.train_file_list{exclude_song_ids(i)});
                end
                fclose(fid);
                fprintf('Saved files to be excluded (%i) to %s\n', ...
                    length(exclude_song_ids), obj.exclude_songs_fln);
            end
        end
        
        
        function make_feats_per_bar(obj, whole_note_div)
            if exist(obj.train_lab_fln, 'file')
                fprintf('    Found %i files in %s\n', length(obj.train_file_list), ...
                    obj.train_lab_fln);
                dataPerBar = [];
                for iDim =1:obj.feature.feat_dim
                    % organise features into bar position grid
                    Output = Data.extract_bars_from_feature(obj.train_file_list, ...
                        obj.feature.feat_type{iDim}, whole_note_div, ...
                        obj.feature.frame_length, obj.pattern_size, 1);
                    % compute mean feature of each bar position cell
                    dataPerBar = [dataPerBar, cellfun(@mean, Output.dataPerBar)];
                end
                % save data to object
                obj.bar2file = Output.bar2file;
                obj.data_per_bar = dataPerBar;
                obj.file_2_meter = Output.file2meter;
                obj.bar_2_meter = Output.file2meter(obj.bar2file, :);
                % save features organised by bars positions to file
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
        
        function [ca_fln, clust_trans_fln] = do_clustering(obj, n_clusters, ...
                pattern_scope, varargin)
            % Clusters the bars using a kmeans algorithm
            %
            % ------------------------------------------------------------------------
            % INPUT parameters:
            % n_clusters         number of clusters
            % pattern_scope      can be either 'bar' or 'song'. With 'song',
            %                    all bars of one song are averaged into one
            %                    song_pattern and these are then clustered
            % 'meters'           [2 x num_meters] meters that are expected
            %                    to be in the data
            % 'meter_names'      cell array [num_meters x 1]: assigns a name
            %                    for each meter
            % 'save_pattern_fig' [default=1] save pattern plot
            % 'plotting_path'    [default='/tmp/']
            %
            % OUTPUT parameters:
            % runtime           runtime (excluding training) in minutes
            %
            %
            % 06.09.2012 by Florian Krebs
            %
            % changelog:
            % 03.01.2013 modularize code into subfunctions
            % 29.01.2015 added config file name input parameter
            % ------------------------------------------------------------------------
            % n_clusters
            % type = 'bar' : cluster data_per_bar
            % type = 'song': cluster data_per_song
            % fig_store_flag: stores a figure with plots of clustering,
            % else no
            % bad_meters    : [1 x nMeters] numerator of meters to ignore
            %                   (e.g. [3, 8, 9])
            % -------------------------------------------------------------
            % parse arguments
            p = inputParser;
            % set defaults
            default_scope = 'song';
            valid_scope = {'bar','song'};
            check_scope = @(x) any(validatestring(x, valid_scope));
            default_plotting_path = '/tmp/';
            % add inputs
            addRequired(p, 'obj', @isobject);
            addRequired(p, 'n_clusters', @isnumeric);
            addOptional(p, 'pattern_scope', default_scope, check_scope);
            addParameter(p, 'meter_names', '', @iscell);
            addParameter(p, 'meters', -1, @isnumeric);
            addParameter(p, 'save_pattern_fig', 1, @isnumeric);
            addParameter(p, 'plotting_path', default_plotting_path, @ischar);
            parse(p, obj, n_clusters, varargin{:});
            % -------------------------------------------------------------
            obj.n_clusters = n_clusters;
            if strcmpi(pattern_scope, 'bar')
                S = obj.data_per_bar;
                meter_per_item = obj.bar_2_meter;
            else
                S = obj.data_per_song;
                meter_per_item = obj.file_2_meter;
            end
            % check if meter found in the data corresponds to meter given
            % in the input arguments
            meter_data = unique(meter_per_item, 'rows');
            if meters == -1
                meters = meter_data;
                % create meter_names from time signature
                for i_meter=1:size(meters, 2)
                    meter_names{i_meter} = [num2str(meters(1, i_meter)), ...
                        '-', num2str(meters(2, i_meter))];
                end
            else
                same_num_meters = (size(meter_data, 2) == size(meters, 2));
                same_content = ismember(meter_data', meters', 'rows');
                if ~(same_num_meters && same_content)
                    error(['ERROR RhythmCluster.do_clustering: Number of ',...
                        'meters in data does not match number of meters', ...
                        'specified in the function input argument']);
                end 
            end
            meters = unique(meter_per_item, 'rows');
            if size(meters, 1) == 1 % only one meter
                n_items_per_meter = size(meter_per_item, 1);
            else % count items per meter
                n_items_per_meter = hist(meter_per_item(:, 1), ...
                    sort(meters(:, 1), 'ascend'));
            end
            % Ajay/Andre: distribute patterns equally among meters
            % n_clusters_per_meter = ceil(n_clusters/size(meters,1))*ones(1,size(meters,1));
            
            % Florian: distribute patterns among meters according to the
            % amount of data we have per meter
            clusters_per_meter = n_items_per_meter * obj.n_clusters ...
                / sum(n_items_per_meter);
            n_clusters_per_meter = ceil(clusters_per_meter);
            while sum(n_clusters_per_meter) > obj.n_clusters % check because of ceil
                % find most crowded cluster
                overhead = clusters_per_meter - n_clusters_per_meter + 1;
                [~, idx] = min(overhead);
                n_clusters_per_meter(idx) = n_clusters_per_meter(idx) - 1;
            end
            % normalise feature to treat them all equally when clustering
            bar_pos = size(S, 2) / obj.feature.feat_dim;
            for i_dim = 1:obj.feature.feat_dim
                vals = S(:, (i_dim-1)*bar_pos+1:bar_pos*i_dim);
                vals = vals - nanmean(vals(:));
                vals = vals / nanstd(vals(:));
                S(:, (i_dim-1)*bar_pos+1:bar_pos*i_dim) = vals;
            end
            opts = statset('MaxIter', 200);
            % cluster different meter separately
            ctrs = zeros(obj.n_clusters, size(S, 2));
            cidx = zeros(size(S, 1), 1);
            p = 1;
            % replace nans because the MATLAB kmeans ignores all datapoints
            % that contain nans
            S(isnan(S)) = -999;
            obj.rhythm2meter = zeros(obj.n_clusters, 2);
            for iMeter=1:size(meters, 1)
                idx_i = (meter_per_item(:, 1) == meters(iMeter, 1)) & ...
                    (meter_per_item(:, 2) == meters(iMeter, 2));
                if n_clusters_per_meter(iMeter) > 1 % more than one cluster per meter -> kmeans
                    [cidx(idx_i), ctrs(p:p+n_clusters_per_meter(iMeter)-1, ...
                        :)] = kmeans(S(idx_i, :), ...
                        n_clusters_per_meter(iMeter), 'Distance', ...
                        'sqEuclidean', 'Replicates', 5, 'Options', opts);
                    cidx(idx_i) = cidx(idx_i) + p - 1;
                    obj.rhythm2meter(p:p+n_clusters_per_meter(iMeter)-1, :) ...
                        = repmat(meters(iMeter, :), ...
                        n_clusters_per_meter(iMeter), 1);
                    p=p+n_clusters_per_meter(iMeter);
                else % only one item per meter -> no kmeans necessary
                    ctrs(p, :) = mean(S(idx_i, :));
                    cidx(idx_i) = p;
                    obj.rhythm2meter(p, :) = meters(iMeter, :);
                    p=p+1;
                end
            end
            % reintroduce nans
            ctrs(ctrs==-999) = nan;
            if save_pattern_fig
                % plot patterns and save plot to png file
                obj.plot_patterns(cidx, ctrs, bar_pos, pattern_scope, ...
                    plotting_path);
            end
            % save cluster assignments
            if strcmpi(pattern_scope, 'bar')
                % use cidx directly from the kmeans clustering
                obj.clusters_fln = fullfile(obj.data_save_path, ['ca-', ...
                    obj.dataset, '-', num2str(obj.feature.feat_dim), ...
                    'd-', num2str(obj.n_clusters), 'R-kmeans.mat']);
            else % assign the cluster idx of the song to each bar which
                 % belongs to that song
                % read index of valid songs
                if exist(obj.exclude_songs_fln, 'file')
                    % TODO: Why read this from file? Can it be saved within
                    % the object?
                    fid = fopen(obj.exclude_songs_fln, 'r');
                    exclude_songs = textscan(fid, '%s');
                    fclose(fid);
                    exclude_songs = find(ismember(obj.train_file_list, ...
                        exclude_songs{1}));
                else
                    exclude_songs = [];
                end
                fileCounter = 0;
                bar2rhythm = [];
                % count bars for each file
                [file2nBars, ~] = hist(obj.bar2file, ...
                    1:length(obj.train_file_list));
                for iFile = 1:length(obj.train_file_list)
                    if ismember(iFile, exclude_songs), continue; end
                    fileCounter = fileCounter + 1;
                    % determine number of bars
                    if file2nBars(iFile) > 0
                        patternId = cidx(fileCounter);
                        bar2rhythm = [bar2rhythm; ...
                            ones(file2nBars(iFile), 1) * patternId];
                    end
                end
                cidx = bar2rhythm;
                obj.clusters_fln = fullfile(obj.data_save_path, ['ca-', ...
                    obj.dataset, '-', num2str(obj.feature.feat_dim), 'd-', ...
                    num2str(obj.n_clusters),'R-kmeans-songs.mat']);
            end
            % make up a name for each newly created rhythm pattern. Ishould
            % contain both the meter and the id of the pattern
            for i_R = 1:obj.n_clusters
                meter_id_of_pattern_i_R = ismember(meters', ...
                    obj.rhythm2meter(i_R, :), 'rows');
                obj.rhythm_names{i_R} = [obj.dataset, '-', ...
                    meter_names{meter_id_of_pattern_i_R}, ...
                    '_', num2str(i_R)];
            end
            obj.bar_2_cluster = cidx;
            obj.file_2_nBars = file2nBars;
            % save all variables related to a cluster assignment to file
            obj.save_cluster_alignment_file();
            ca_fln = obj.clusters_fln;
            clust_trans_fln = obj.compute_cluster_transitions();
        end
        
        function clust_trans_fln = compute_cluster_transitions(obj)
            A = zeros(obj.n_clusters, obj.n_clusters);
            for iFile=1:length(obj.train_file_list)
                bars = find(obj.bar2file==iFile);
                for iBar=bars(1:end-1)
                    A(obj.bar_2_cluster(iBar), obj.bar_2_cluster(iBar+1)) = ...
                        A(obj.bar_2_cluster(iBar), obj.bar_2_cluster(iBar+1)) + 1;
                end
            end
            % normalise transition matrix
            A = bsxfun(@rdivide, A , sum(A , 2));
            obj.cluster_transition_matrix = A;
            % obj.cluster_transition_matrix = eye(size(A));
            obj.cluster_transitions_fln = fullfile(obj.data_save_path, ...
                ['cluster_transitions-', obj.dataset, '-', ...
                num2str(obj.feature.feat_dim), 'd-', ...
                num2str(obj.n_clusters), '.txt']);
            dlmwrite(obj.cluster_transitions_fln, obj.cluster_transition_matrix);
            clust_trans_fln = obj.cluster_transitions_fln;
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
            % bar2rhythm        : [nBars x 1] assigns each bar to a pattern
            % file_2_cluster    : [nFiles x 1] assigns each file to a pattern
            % n_clusters        : number of patterns
            % bar2file          : [nBars x 1] assigns each bar a file id
            % file2nBars        : [nFiles x 1] number of bars per file
            % rhythm_names      : {R x 1} string
            % rhythm2meter      : [R x 2] meter for each pattern
            %
            % 09.07.2013 by Florian Krebs
            % ----------------------------------------------------------------------
            if nargin == 1
                clusterType = 'auto';
            end
            if strcmpi(clusterType, 'auto')
                songClusterIds = load(obj.clusters_fln, '-ascii');
            end
            bar2rhythm = [];
            obj.bar2file = [];
            %             dancestyles = {'ChaCha', 'Jive', 'Quickstep', 'Rumba', 'Samba', 'Tango', ...
            %               'VienneseWaltz', 'Waltz'};
            % check if there are songs that should be excluded
            if isempty(obj.exclude_songs_fln)
                ok_songs = 1:length(obj.train_file_list);
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
                [~, fname, ~] = fileparts(obj.train_file_list{ok_songs(iFile)});
                fprintf('%i) %s\n', iFile, fname);
                [beats, error] = Data.load_annotations_bt(...
                    obj.train_file_list{ok_songs(iFile)}, 'beats');
                if error, error('no beat file found\n'); end
                if size(beats, 2) < 2
                    error('Downbeat annotations missing for %s\n', ...
                        obj.train_file_list{ok_songs(iFile)});
                end
                meter(fileCounter+1) = max(beats(:, 3));
                
                % get pattern id of file
                switch lower(clusterType)
                    case 'meter'
                        patternId = meter(fileCounter+1);
                    case 'dancestyle'
                        style = Data.load_annotations_bt(...
                            obj.train_file_list{ok_songs(iFile)}, 'dancestyle');
                        patternId = find(strcmp(rhythm_names, style));
                        if isempty(patternId)
                            fprintf('Please add %s to the rhythm_names\n', style);
                        end
                        obj.file_2_cluster(iFile) = patternId;
                    case 'auto'
                        patternId = songClusterIds(iFile);
                    case 'rhythm'
                        style = Data.load_annotations_bt(...
                            strrep(obj.train_file_list{ok_songs(iFile)}, ...
                            'audio', 'annotations/rhythm'), 'rhythm');
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
            end
            obj.n_clusters = max(bar2rhythm);
            obj.rhythm2meter = zeros(obj.n_clusters, 2);
            for iR=1:obj.n_clusters
                % find a bar/file that represents rhythm iR
                file_id = obj.bar2file(find(bar2rhythm==iR, 1));
                meter = Data.load_annotations_bt(...
                    strrep(obj.train_file_list{file_id}, 'audio', ...
                    'annotations/meter'), 'meter');
                % TODO: what to do if meter of training data does not match
                % meter of system ?
                % BUG HERE: The meters have to be ordered in the increasing
                % order, fails otherwise!
                if strcmp(obj.pattern_size, 'bar')
                    obj.rhythm2meter(iR, :) = meter;
                elseif strcmp(obj.pattern_size, 'beat')
                    obj.rhythm2meter(iR, :) = [1, 4];
                else
                    error('Meter of training data is not supported by the system')
                end
            end
            obj.bar_2_cluster = bar2rhythm;
            obj.clusters_fln = fullfile(obj.data_save_path, ['ca-', obj.dataset, ...
                '-', num2str(obj.feature.feat_dim), 'd-', num2str(obj.n_clusters), ...
                'R-', clusterType, '.mat']);
            if ~exist('rhythm_names', 'var')
                for i = unique(bar2rhythm(:))'
                    rhythm_names{i} = [clusterType, num2str(i)];
                end
            end
            obj.rhythm_names = rhythm_names;
            obj.file_2_nBars = nBars;
            obj.save_cluster_alignment_file();
        end
    end
    
    methods (Access=protected)
        function [] = save_cluster_alignment_file(obj)
            rhythm2meter = obj.rhythm2meter;
            bar2file = obj.bar2file;
            bar2rhythm = obj.bar_2_cluster;
            rhythm_names = obj.rhythm_names;
            file2nBars = obj.file_2_nBars;
            save(obj.clusters_fln, '-v7.3', 'bar2rhythm', 'bar2file', ...
                'file2nBars', 'rhythm_names', 'rhythm2meter');
            fprintf('    Saved bar2rhythm, bar2file, file2nBars, rhythm_names, ');
            fprintf('rhythm2meter to %s\n', obj.clusters_fln);
        end
        
        function [] = plot_patterns(cidx, ctrs, bar_pos, pattern_scope, ...
                plotting_path)
            plot_cols = ceil(sqrt(obj.n_clusters));
                h = figure( 'Visible','off');
                set(h, 'Position', [100 100 obj.n_clusters*100 obj.n_clusters*100]);
                items_per_cluster = hist(cidx, obj.n_clusters);
                col = hsv(obj.feature.feat_dim);
                for c = 1:obj.n_clusters
                    subplot(ceil(obj.n_clusters/plot_cols), plot_cols, c)
                    hold on
                    for fdim = 1:obj.feature.feat_dim
                        data = ctrs(c, (fdim-1)*bar_pos+1:fdim*bar_pos);
                        data = data - min(data);
                        data = data / max(data);
                        data = data + fdim;
                        stairs([data, data(end)], 'Color', col(fdim, :));
                    end
                    if obj.feature.feat_dim == 1
                        y_label = obj.feature.feat_type{1};
                    else
                        y_label = sprintf('Bottom: %s', ...
                            strrep(obj.feature.feat_type{1}, '_', '\_'));
                    end
                    ylabel(sprintf('%s', y_label));
                    xlabel('bar position')
                    title(sprintf('cluster %i (%i %s)', c, ...
                        items_per_cluster(c), pattern_scope));
                    xlim([1 length(data)])
                end
                outfile = fullfile(plotting_path, ['patterns-', ...
                    obj.dataset, '-kmeans-', pattern_scope, '-', ...
                    num2str(obj.n_clusters), '.png']);
                fprintf('    Writing patterns to %s\n', outfile);
                % save to png
                print(h, outfile, '-dpng');
        end
    end
end

