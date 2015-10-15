classdef RhythmCluster < handle
% This class contains functions to cluster bars of a training set according
% to features (kmeans) or labels (meter, rhythm).
    
    properties
        feature             % feature object
        data                % Data object on which clustering is 
                            %  performed
        data_save_path      % path where cluster ids per bar are stored
        n_clusters          % number of clusters
        bar2file            % [nBars x 1] vector
        bar2cluster         % [nBars x 1] vector
        file2meter          % [nFiles x 2]
        rhythm2meter        % [nRhythms x 2]
        rhythm2nbeats       % number of beats per bar [nRhythms x 1]
        rhythm_names        % {R x 1} strings
        pr                  % transition probability matrix of the rhythmic 
                            % patterns [R x R]
        pattern_prior       % prior probability [R x 1]
    end
    
    methods
        function obj = RhythmCluster(train_data, data_save_path)
            %  obj = RhythmCluster(train_data, data_save_path)
            %  Construct Rhythmcluster object
            % -------------------------------------------------------------
            %INPUT parameter:
            % train_data                : Data object
            % data_save_path            : path where clusterings, plots,
            %                               etc. are saved
            %
            % 09.04.2013 by Florian Krebs
            % -------------------------------------------------------------
            % Store some required information:
            obj.feature = train_data.feature;
            obj.data = train_data;
            if exist('data_save_path', 'var')
                obj.data_save_path = data_save_path;
            end
        end
        
        
        function [] = cluster_from_features(obj, n_clusters, varargin)
            % [] = cluster_from_features(obj, features, n_clusters, ...
            %    varargin)
            % Clusters the bars using a kmeans algorithm
            %
            % -------------------------------------------------------------
            % INPUT parameters:          
            % n_clusters         number of clusters
            % 'dist_cluster'     can be either 'data' (clusters per meter assigned
            %                    based on amount of data per meter) or 'equal' 
            %                    (each meter has the same number of clusters): 
            % 'pattern_scope'    can be either 'bar' or 'song'. With 'song',
            %                    all bars of one song are averaged into one
            %                    song_pattern and these are then clustered
            % 'meters'           [num_meters x 2] meters that are expected
            %                    to be in the data
            % 'meter_names'      cell array [num_meters x 1]: assigns a name
            %                    for each meter
            % 'save_pattern_fig' [default=1] save pattern plot
            % 'plotting_path'    [default='/tmp/']
            %
            % 06.09.2012 by Florian Krebs
            %
            % changelog:
            % 03.01.2013 modularize code into subfunctions
            % 29.01.2015 added config file name input parameter
            % ------------------------------------------------------------------------
            % parse arguments
            parser = inputParser;
            % set defaults
            default_scope = 'bar';
            valid_scope = {'bar', 'song'};
            check_scope = @(x) any(validatestring(x, valid_scope));
            valid_dist_clust = {'data', 'equal'};
            default_dist_clust = valid_dist_clust{1};
            check_dist_clust = @(x) any(validatestring(x, valid_dist_clust));
            default_plotting_path = '/tmp/';
            % add inputs
            addRequired(parser, 'obj', @isobject);
            addRequired(parser, 'n_clusters', @isnumeric);
            addOptional(parser, 'dist_cluster', default_dist_clust, ...
                check_dist_clust);
            addOptional(parser, 'pattern_scope', default_scope, ...
                check_scope);
            addParamValue(parser, 'meter_names', '', @iscell);
            addParamValue(parser, 'meters', -1, @isnumeric);
            addParamValue(parser, 'save_pattern_fig', 1, @isnumeric);
            addParamValue(parser, 'plotting_path', default_plotting_path, ...
                @ischar);
            parse(parser, obj, n_clusters, ...
                varargin{:});
            % -------------------------------------------------------------
            obj.n_clusters = n_clusters;
            pattern_scope = parser.Results.pattern_scope;
            dist_cluster = parser.Results.dist_cluster;
            % summarise features for clustering
            S = cellfun(@mean, obj.data.features_per_bar);
            bar_pos = size(S, 2);
            if strcmpi(pattern_scope, 'bar')
                meter_per_item = obj.data.meters(obj.data.bar2file, :);
                n_beats_per_item = obj.data.beats_per_bar(...
                    obj.data.bar2file, :);
            elseif strcmpi(pattern_scope, 'song')
                % summarise features of one song
                S = obj.average_feats_per_song(S, obj.data.bar2file, ...
                    length(obj.data.file_list));
                meter_per_item = obj.data.meters;
                n_beats_per_item = obj.data.beats_per_bar;
            end
            % normalise features to zero mean and unit std before
            % clustering
            for i_dim = 1:obj.data.feature.feat_dim
                vals = S(:, :, i_dim);
                vals = vals - nanmean(vals(:));
                vals = vals / nanstd(vals(:));
                S(:, :, i_dim) = vals;
            end
            % number of items to be clustered
            n_items = size(S, 1);
            % cluster dimensions. These are number of bar positions times
            % the feature dimension
            n_feat_dims_clustering = size(S, 2) * size(S, 3);
            S = reshape(S, [n_items, n_feat_dims_clustering]);
            [meters, idx, ~] = unique(meter_per_item, 'rows');
            if size(meters, 1) > obj.n_clusters
                fprintf(['    WARNING: need %i patterns to describe all time', ...
                    'signatures in the data\n'], size(meters, 1));
                obj.n_clusters = size(meters, 1);
            end
            beats_per_bar = n_beats_per_item(idx);
            if parser.Results.meters == -1
                % no meters given as input: use all meters that were found
                % in the data
                % check if meter found in the data corresponds to meter given
                % in the input arguments [num_meters x 2]
                assert(size(meters, 2) == 2, 'Meters have wrong dimension.')
                 % create meter_names from time signature
                meter_names = cell(size(meters, 1), 1);
                for i_meter=1:size(meters, 1)
                    meter_names{i_meter} = [num2str(meters(i_meter, 1)), ...
                        '-', num2str(meters(i_meter, 2))];
                end
            else
                meter_names = parser.Results.meter_names;
                same_num_meters = (size(meters, 1) == ...
                    size(parser.Results.meters, 1));
                same_content = ismember(meters, parser.Results.meters, ...
                    'rows');
                assert(same_num_meters && same_content, ['ERROR ', ...
                    'RhythmCluster.do_clustering: Number of ',...
                    'meters in data does not match number of meters', ...
                    'specified in the function input argument']);
            end          
            if size(meters, 1) == 1 % only one meter
                n_items_per_meter = size(meter_per_item, 1);
            else % count items per meter
                n_items_per_meter = hist(meter_per_item(:, 1), ...
                    sort(meters(:, 1), 'ascend'));
            end
            if strcmp(dist_cluster,'equal')
                % Ajay/Andre: distribute patterns equally among meters
                n_clusters_per_meter = ceil(n_clusters/size(meters,1))*...
                ones(1,size(meters,1));
            else
                % Or: distribute patterns among meters according to the
                % amount of data we have per meter
                clusters_per_meter = n_items_per_meter * obj.n_clusters ...
                    / sum(n_items_per_meter);
                n_clusters_per_meter = ceil(clusters_per_meter);
            end
            while sum(n_clusters_per_meter) > obj.n_clusters % check because of ceil
                % find least crowded cluster
                overhead = clusters_per_meter - n_clusters_per_meter + 1;
                % keep at least one cluster per meter
                overhead(n_clusters_per_meter==1) = 10;
                [~, idx] = min(overhead);
                n_clusters_per_meter(idx) = n_clusters_per_meter(idx) - 1;
            end
            opts = statset('MaxIter', 200);
            % cluster different meter separately
            ctrs = zeros(obj.n_clusters, size(S, 2));
            cidx = zeros(size(S, 1), 1);
            p = 1;
            % replace nans by a high number because the MATLAB kmeans
            % ignores all datapoints that contain nans. A high number
            % prevents bars of different length obtain the same cluster id
            high_distance = -99999999;
            S(isnan(S)) = high_distance;
            obj.rhythm2meter = zeros(obj.n_clusters, 2);
            obj.rhythm2nbeats = zeros(obj.n_clusters, 1);
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
                    obj.rhythm2nbeats(p:p+n_clusters_per_meter(iMeter)-1) = ...
                        beats_per_bar(iMeter);
                    p=p+n_clusters_per_meter(iMeter);
                else % only one item per meter -> no kmeans necessary
                    ctrs(p, :) = mean(S(idx_i, :));
                    cidx(idx_i) = p;
                    obj.rhythm2meter(p, :) = meters(iMeter, :);
                    obj.rhythm2nbeats(p) = beats_per_bar(iMeter);
                    p=p+1;
                end
            end
            % reintroduce nans
            ctrs(ctrs==high_distance) = nan;
            if parser.Results.save_pattern_fig
                % plot patterns and save plot to png file
                obj.plot_patterns(cidx, ctrs, bar_pos, pattern_scope, ...
                    parser.Results.plotting_path);
            end
            % save cluster assignments
            if strcmpi(pattern_scope, 'bar')
                % use cidx directly from the kmeans clustering
                cluster_type_string = 'kmeans-bars';
                bar2rhythm = cidx;
            else % assign the cluster idx of the song to each bar which
                % belongs to that song
                cluster_type_string = 'kmeans-songs';
                % read index of valid songs
                bar2rhythm = zeros(length(obj.data.bar2file), 1);
                ok_file_counter = 0;
                for iFile = 1:length(obj.data.file_list)
                    ok_file_counter = ok_file_counter + 1;
                    % determine number of bars
                    if obj.data.n_bars(iFile) > 0
                        patternId = cidx(ok_file_counter);
                        bar2rhythm(obj.data.bar2file == iFile) = patternId;
                    end
                end
            end
            % make up a name for each newly created rhythm pattern. Ishould
            % contain both the meter and the id of the pattern
            for i_R = 1:obj.n_clusters
                meter_id_of_pattern_i_R = ismember(meters, ...
                    obj.rhythm2meter(i_R, :), 'rows');
                obj.rhythm_names{i_R} = [obj.data.dataset, '-', ...
                    meter_names{meter_id_of_pattern_i_R}, ...
                    '_', num2str(i_R)];
            end
            obj.bar2cluster = bar2rhythm;
            % create rhythm transitions
            obj.compute_cluster_transitions();
            % save all variables related to a cluster assignment to file
            if ~isempty(obj.data_save_path)
                obj.save_cluster_alignment_file(cluster_type_string);
            end
        end
        
        function [] = compute_cluster_transitions(obj)
            % [] = compute_cluster_transitions(obj)
            %   Count transitions between clusters and set up a transition
            %   probability matrix obj.pr.
            % ------------------------------------------------------------
            % 11.08.2015 by Florian Krebs
            % ------------------------------------------------------------
            if obj.n_clusters > 1
                pattern_prior = zeros(obj.n_clusters, 1);
                A = zeros(obj.n_clusters, obj.n_clusters);
                for iFile=1:max(obj.data.bar2file)
                    bars = find(obj.data.bar2file==iFile);
                    if ~isempty(bars)
                        for iBar=bars(1:end-1)'
                            pattern_i = obj.bar2cluster(iBar);
                            A(pattern_i, obj.bar2cluster(iBar+1)) = ...
                                A(pattern_i, obj.bar2cluster(iBar+1)) + 1;
                            pattern_prior(pattern_i) = pattern_prior(pattern_i) + 1;
                        end
                        pattern_prior(obj.bar2cluster(bars(end))) = ...
                            pattern_prior(obj.bar2cluster(bars(end))) + 1;
                    end
                end
                % normalise transition matrix
                obj.pr = bsxfun(@rdivide, A , sum(A , 2));
                obj.pattern_prior = pattern_prior / sum(pattern_prior);
            else % only one pattern
                obj.pr = 1;
                obj.pattern_prior = 1;
            end
        end
        
        
        function [] = cluster_from_labels(obj, clusterType,  ...
                rhythm_anns, rhythm_names)
            % data = cluster_from_labels(obj, data, clusterType,  ...
            %    rhythm_anns, rhythm_names)
            %   Cluster the bars from data according to given labels.
            % ----------------------------------------------------------------------
            %INPUT parameter:
            % data          : filename of labfile (e.g., 'boeck.lab' or 'ballroom.lab').
            %                       a labfile is a textfile with paths to files that are analyzed
            % clusterType         : {'meter', 'rhythm', 'none'}
            %                   'meter': bars are clustered according to the meter (functions reads .meter file);
            %                   'rhythm', according to the genre (functions reads .rhythm file))
            %                   'none' : put all bars into one single
            %                   cluster
            % rhythm_names      : cell array of strings
            %
            %OUTPUT parameter:
            % data              : Data instance with updated properties
            %                       (bar2cluster, pr, n_clusters,
            %                       rhythm2meter, rhythm_names)
            %
            % 04.08.2015 Florian Krebs: modify Data instance
            % 09.07.2013 by Florian Krebs
            % ----------------------------------------------------------------------
            bar2rhythm = zeros(sum(obj.data.n_bars), 1);
            obj.bar2file = zeros(sum(obj.data.n_bars), 1);
            % unique meters present in the data [T x 2]
            meters = unique(obj.data.meters, 'rows');
            bar_counter = 0;
            for iFile = 1:length(obj.data.file_list)
                beats = obj.data.beats{iFile};
                if size(beats, 2) < 2
                    error('Downbeat annotations missing for %s\n', ...
                        obj.data.file_list{iFile});
                end
                meter = obj.data.meters(iFile, :);
                beats_per_bar = obj.data.beats_per_bar(iFile);
                % So far, we assume that labels are only given on the song
                % level, which means we assign the same cluster id to all
                % bars of the file
                switch lower(clusterType)
                    case 'meter'
                        % get the cluster id of the meter
                        cluster_id = find(ismember(meters, ...
                            meter, 'rows'));
                    case 'rhythm'
                        cluster_id = find(strcmp(rhythm_names, ...
                            rhythm_anns{iFile}));
                        if isempty(cluster_id)
                            fprintf('Please add %s to the rhythm_names\n', ...
                                style);
                        end
                    case 'none'
                        cluster_id = 1;
                end
                if strcmp(obj.data.pattern_size, 'bar')
                    obj.rhythm2meter(cluster_id, :) = meter;
                    obj.rhythm2nbeats(cluster_id) = beats_per_bar;
                elseif strcmp(obj.data.pattern_size, 'beat')
                    obj.rhythm2meter(cluster_id, :) = [1, 4];
                    obj.rhythm2nbeats(cluster_id) = 1;
                end
                bar_idx = bar_counter+1:bar_counter+obj.data.n_bars(iFile);
                bar2rhythm(bar_idx) = cluster_id;
                obj.bar2file(bar_idx) = iFile;
                bar_counter = bar_counter + obj.data.n_bars(iFile);
            end
            obj.n_clusters = max(bar2rhythm);
            obj.bar2cluster = bar2rhythm;
            if ~exist('rhythm_names', 'var')
                for i = unique(bar2rhythm(:))'
                    rhythm_names{i} = [clusterType, num2str(i)];
                end
            end
            obj.rhythm2nbeats = obj.rhythm2nbeats(:);
            obj.rhythm_names = rhythm_names;
            obj.compute_cluster_transitions();
            if ~isempty(obj.data_save_path)
                obj.save_cluster_alignment_file(clusterType);
            end
        end
    end
    
    methods (Access=protected)
        function [] = save_cluster_alignment_file(obj, cluster_type_string)
            rhythm2meter = obj.rhythm2meter;
            bar2file = obj.bar2file;
            bar2rhythm = obj.bar2cluster;
            rhythm_names = obj.rhythm_names;
            pr = obj.pr;
            pattern_prior = obj.pattern_prior;
            clusters_fln = fullfile(obj.data_save_path, ['ca-', ...
                obj.dataset, '-', num2str(obj.feature.feat_dim), 'd-', ...
                num2str(obj.n_clusters), 'R-', cluster_type_string, ...
                '.mat']);
            save(clusters_fln, '-v7.3', 'bar2rhythm', 'bar2file', ...
                'rhythm_names', 'rhythm2meter', 'pr', 'pattern_prior');
            fprintf('    Saved clustering data to %s\n', clusters_fln);
        end
        
        function [] = plot_patterns(obj, cidx, ctrs, bar_pos, pattern_scope, ...
                plotting_path)
            plot_cols = ceil(sqrt(obj.n_clusters));
            h = figure( 'Visible','off');
            set(h, 'Position', ...
                [100 100 obj.n_clusters*100 obj.n_clusters*100]);
            items_per_cluster = hist(cidx, obj.n_clusters);
            col = hsv(obj.feature.feat_dim);
            for c = 1:obj.n_clusters
                subplot(ceil(obj.n_clusters/plot_cols), plot_cols, c)
                hold on
                for fdim = 1:obj.feature.feat_dim
                    pattern = ctrs(c, (fdim-1)*bar_pos+1:fdim*bar_pos);
                    pattern = pattern - min(pattern);
                    pattern = pattern / max(pattern);
                    pattern = pattern + fdim;
                    stairs([pattern, pattern(end)], 'Color', col(fdim, :));
                end
                if obj.feature.feat_dim == 1
                    y_label = obj.feature.feat_type{1};
                else
                    y_label = sprintf('Bottom: %s', ...
                        strrep(obj.feature.feat_type{1}, '_', '\_'));
                end
                ylabel(sprintf('%s', y_label));
                xlabel('bar position')
                title(sprintf('cluster %i (%i %ss)', c, ...
                    items_per_cluster(c), pattern_scope));
                xlim([1 length(pattern)])
            end
            outfile = fullfile(plotting_path, ['patterns-', ...
                obj.data.dataset, '-kmeans-', pattern_scope, '-', ...
                num2str(obj.n_clusters), '.png']);
            fprintf('    Writing patterns to %s\n', outfile);
            % save to png
            print(h, outfile, '-dpng');
        end
    end
    
    methods (Static)
        
        function data_per_song = average_feats_per_song(data_per_bar, ...
                bar2file, n_files)
            % data_per_song = average_feats_per_song(data_per_bar, ...
            %     bar2file, n_files)
            % Computes mean of features for all bar positions within a song
            %
            %INPUT parameter:
            %   data_per_bar : [nBars, nPos, nDim]
            %OUTPUT parameter: 
            %   data_per_song [nSongs x (feat_dim*bar_grid)]
            % ------------------------------------------------------------
            % compute mean per song
            data_per_song = NaN(n_files, ...
                size(data_per_bar, 2), size(data_per_bar, 3));
            for iPos=1:size(data_per_bar, 2) % loop over barPos
                for iDim=1:size(data_per_bar, 3) % loop over FeatDim
                    data_per_song(:, iPos, iDim) = accumarray(bar2file(:), ...
                        data_per_bar(:, iPos, iDim), [], @mean);
                end
            end
        end
    end
end
