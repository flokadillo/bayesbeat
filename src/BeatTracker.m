classdef BeatTracker < handle
    % Beat tracker Class
    properties
        model                   % probabilistic model
        feature
        train_data
        test_data
        sim_dir                 % directory where results are saved
        init_model_fln          % fln of initial model to start with
        Params                  % parameters from config file
    end
    
    methods
        
        function obj = BeatTracker(Params)
            % parse parameters and set defaults
            obj.parse_params(Params);
            % load or create probabilistic model
            obj.init_model();
            % initialise test data
            obj.init_test_data();
        end
        
        function init_model(obj)
            if isfield(obj.Params, 'model_fln') && ~isempty(obj.Params.model_fln)
                if exist(obj.Params.model_fln, 'file')
                    c = load(obj.Params.model_fln);
                    fields = fieldnames(c);
                    obj.model = c.(fields{1});
                    %                     obj.model = obj.convert_to_new_model_format(obj.model);
                    obj.init_model_fln = obj.Params.model_fln;
                    obj.feature = Feature(obj.model.obs_model.feat_type, ...
                        obj.model.frame_length);
                    if isfield(obj.Params, 'use_mex_viterbi')
                        obj.model.use_mex_viterbi = obj.Params.use_mex_viterbi;
                    end
                    fprintf('\n* Loading model from %s\n', obj.Params.model_fln);
                else
                    error('Model file %s not found', obj.Params.model_fln);
                end
            else
                obj.feature = Feature(obj.Params.feat_type, ...
                    obj.Params.frame_length);
                % initialise training data to see how many pattern states
                % we need and which time signatures have to be modelled.
                % the data is collected and clustered
                obj.init_train_data();
                % re-format tempo ranges to have one value per pattern
                R = obj.train_data.clustering.n_clusters;
                if (length(obj.Params.min_tempo_bpm) == 1) && (R > 1)
                    obj.Params.min_tempo_bpm = repmat(obj.Params.min_tempo_bpm, ...
                        R, 1);
                end
                if (length(obj.Params.max_tempo_bpm) == 1) && (R > 1)
                    obj.Params.max_tempo_bpm = repmat(obj.Params.max_tempo_bpm, ...
                        R, 1);
                end
                if obj.Params.learn_tempo_ranges
                    % get tempo ranges from data for each file
                    [tempo_min_per_cluster, tempo_max_per_cluster] = ...
                        obj.train_data.get_tempo_per_cluster(...
                        obj.Params.tempo_outlier_percentile);
                    % find min/max for each pattern
                    tempo_min_per_cluster = min(tempo_min_per_cluster)';
                    tempo_max_per_cluster = max(tempo_max_per_cluster)';
                    % restrict ranges and store modified tempo ranges
                    tempo_above_min = tempo_min_per_cluster > ...
                        obj.Params.min_tempo_bpm(:);
                    tempo_below_max = tempo_max_per_cluster < ...
                        obj.Params.max_tempo_bpm(:);
                    obj.Params.min_tempo_bpm(tempo_above_min) = ...
                        tempo_min_per_cluster(tempo_above_min);
                    obj.Params.max_tempo_bpm(tempo_below_max) = ...
                        tempo_max_per_cluster(tempo_below_max);
                    if obj.Params.same_tempo_per_meter
                        [meters, ~, meter_id_from_pattern] = ...
                            unique(obj.train_data.clustering.rhythm2meter, ...
                            'rows');
                        for i_meter = 1:size(meters, 1)
                            % use min and max of all patterns that belong
                            % to same meter
                            is_i_meter = (meter_id_from_pattern == i_meter);
                            obj.Params.min_tempo_bpm(is_i_meter) = ...
                                min(obj.Params.min_tempo_bpm(is_i_meter));
                            obj.Params.max_tempo_bpm(is_i_meter) = ...
                                max(obj.Params.max_tempo_bpm(is_i_meter));
                        end
                    end
                end
                switch obj.Params.inferenceMethod(1:2)
                    case 'HM'
                        obj.model = BeatTrackingModelHMM(obj.Params, ...
                            obj.train_data.clustering);
                    case 'PF'
                        obj.model = BeatTrackingModelPF(obj.Params, ...
                            obj.train_data.clustering);
                    otherwise
                        error('BeatTracker.init_model: inference method %s not known', ...
                            obj.Params.inferenceMethod);
                end
            end
        end
        
        function init_train_data(obj)
            % set up the training data and load or compute the cluster
            % assignments of the bars/beats
            fprintf('* Set up training data ...\n');
            % check if features are already saved
            if isfield(obj.Params, 'stored_train_data_fln') && ...
                    exist(obj.Params.stored_train_data_fln, 'file') && ...
                    obj.Params.load_training_data
                fprintf('    Loading features from %s\n', ...
                    obj.Params.stored_train_data_fln);
                load(obj.Params.stored_train_data_fln, 'data');
                obj.train_data = data;
            else
                obj.train_data = Data(obj.Params.trainLab, ...
                    obj.Params.feat_type, obj.Params.frame_length, ...
                    obj.Params.pattern_size);
                if ~strcmp(obj.Params.observationModelType, 'RNN')
                    obj.train_data.organise_feats_into_bars(...
                        obj.Params.whole_note_div);
                end
                % process silence data
                if obj.Params.use_silence_state
                    fid = fopen(obj.Params.silence_lab, 'r');
                    silence_files = textscan(fid, '%s\n');
                    silence_files = silence_files{1};
                    fclose(fid);
                    obj.train_data.feats_silence = [];
                    for iFile=1:length(silence_files)
                        obj.train_data.feats_silence = ...
                            [obj.train_data.feats_silence;  ...
                            obj.train_data.feature.load_feature(...
                            silence_files{iFile})];
                    end
                end
                % save extracted training data
                if obj.Params.store_training_data
                    data = obj.train_data;
                    save(obj.Params.stored_train_data_fln, 'data');
                end
            end
            % Check if cluster assignment file is given. If yes load
            % cluster assignments from yes, if no compute them.
            if isfield(obj.Params, 'clusterIdFln') && ...
                    exist(obj.Params.clusterIdFln, 'file')
                obj.train_data = obj.train_data.read_pattern_bars(...
                    obj.Params.clusterIdFln, obj.Params.pattern_size);
            else
                if strcmp(obj.Params.observationModelType, 'RNN')
                    obj.train_data.clustering.rhythm2nbeats = 1;
                    obj.train_data.clustering.rhythm2meter = [1, 4];
                    obj.train_data.clustering.rhythm_names = {'rnn'};
                    obj.train_data.clustering.pr = 1;
                    obj.train_data.clustering.n_clusters = 1;
                else
                    fprintf('* Clustering data by %s ...\n', ...
                        obj.Params.cluster_type);
                    if ismember(obj.Params.cluster_type, {'meter', ...
                            'rhythm'})
                        obj.train_data.cluster_from_labels(...
                            obj.Params.cluster_type);
                    elseif strcmp(obj.Params.cluster_type, 'kmeans')
                        if isfield(obj.Params, 'meters') && ...
                                isfield(obj.Params, 'meter_names')
                            obj.train_data.cluster_from_features(...
                                obj.Params.n_clusters, ...
                                'pattern_scope', obj.Params.pattern_size, ...
                                'dist_cluster', obj.Params.dist_cluster, ...
                                'meters', obj.Params.meters, ...
                                'meter_names', obj.Params.meter_names, ...
                                'plotting_path', obj.Params.results_path);  
                        else
                            obj.train_data.cluster_from_features(...
                                obj.Params.n_clusters, ...
                                'pattern_scope', obj.Params.pattern_size, ...
                                'dist_cluster', obj.Params.dist_cluster, ...
                                'plotting_path', obj.Params.results_path);
                        end
                    end
                    fprintf('done\n  %i clusters detected.\n', ...
                        obj.train_data.clustering.n_clusters);
                end
            end
            obj.Params.R = obj.train_data.clustering.n_clusters;
            if isempty(obj.train_data.clustering.pr)
                obj.Params.transition_params.pr = obj.Params.pr;
            else
                obj.Params.transition_params.pr = obj.train_data.clustering.pr;
            end
        end
        
        function init_test_data(obj)
            % create test_data object
            obj.test_data = Data(obj.Params.testLab, obj.Params.feat_type, ...
                obj.Params.frame_length, obj.Params.pattern_size);
        end
        
        function train_model(obj)
            fprintf('* Training model (%s) ...\n',...
                obj.Params.inferenceMethod)
            if isempty(obj.init_model_fln)
                obj.model.train_model(obj.Params.transition_params, ...
                    obj.train_data, obj.Params.whole_note_div, ...
                    obj.Params.observationModelType, obj.Params.results_path);
            end
            if isfield(obj.Params, 'viterbi_learning_iterations') && ...
                    obj.Params.viterbi_learning_iterations > 0
                obj.refine_model(obj.Params.viterbi_learning_iterations);
            end
        end
        
        function constraints = load_constraints(obj, test_file_id)
            for c = 1:length(obj.Params.constraint_type)
                fln = strrep(obj.test_data.file_list{test_file_id}, 'audio', ...
                    ['annotations/', obj.Params.constraint_type{c}]);
                [~, ~, ext] = fileparts(fln);
                fln = strrep(fln, ext, ['.', obj.Params.constraint_type{c}]);
                if strcmp(obj.Params.constraint_type{c}, 'downbeats')
                    data = load(fln);
                    constraints{c} = data(:, 1);
                end
                if strcmp(obj.Params.constraint_type{c}, 'beats')
                    data = load(fln);
                    constraints{c} = data(:, 1);
                end
                if strcmp(obj.Params.constraint_type{c}, 'meter')
                    constraints{c} = Data.load_annotations_bt(fln);
                end
            end
        end
        
        function results = do_inference(obj, test_file_id)
            [~, fname, ~] = fileparts(obj.test_data.file_list{test_file_id});
            fprintf('* Started inference on %s', fname);
            % load feature
            observations = obj.feature.load_feature(...
                obj.test_data.file_list{test_file_id}, ...
                obj.Params.save_features_to_file, ...
                obj.Params.load_features_from_file);
            if isfield(obj.Params, 'constraint_type')
                Constraint.type = obj.Params.constraint_type;
                Constraint.data = obj.load_constraints(test_file_id);
                belief_func = obj.model.make_belief_function(Constraint);
                results = obj.model.do_inference(observations, ...
                    obj.Params.inferenceMethod, fname, belief_func);
            else
                results = obj.model.do_inference(observations, ...
                    obj.Params.inferenceMethod, fname);
            end
        end
        
        function load_model(obj, fln)
            temp = load(fln);
            names = fieldnames(temp);
            obj.model = temp.(names{1});
        end
        
        function [] = save_results(obj, results, save_dir, fname)
            if ~exist(save_dir, 'dir')
                system(['mkdir ', save_dir]);
            end
            if obj.Params.save_beats
                BeatTracker.save_beats(results{1}, fullfile(save_dir, ...
                    [fname, '.beats.txt']));
            end
            if obj.Params.save_downbeats
                BeatTracker.save_downbeats(results{1}, fullfile(save_dir, ...
                    [fname, '.downbeats.txt']));
            end
            if obj.Params.save_median_tempo
                BeatTracker.save_median_tempo(results{2}, fullfile(save_dir, ...
                    [fname, '.bpm.txt']));
            end
            if obj.Params.save_tempo_sequence
                ts = (0:length(results{2})-1) * obj.feature.frame_length;
                BeatTracker.save_tempo_sequence(results{2}, ts, fullfile(save_dir, ...
                    [fname, '.bpm.seq.txt']));
            end
            if obj.Params.save_meter
                BeatTracker.save_meter(results{3}, fullfile(save_dir, ...
                    [fname, '.meter.txt']));
            end
            if obj.Params.save_rhythm
                ts = (0:length(results{4})-1) * obj.feature.frame_length;
                BeatTracker.save_rhythm(results{4}, ...
                    obj.model.state_space.pattern_names, ts, fullfile(save_dir, ...
                    [fname, '.rhythm.txt']));
            end
        end
        
        function parse_params(obj, Params)
            % save parameters
            obj.Params = Params;
            if ~isfield(obj.Params, 'inferenceMethod')
                obj.Params.inferenceMethod = 'HMM_viterbi';
            else
                obj.Params.inferenceMethod = Params.inferenceMethod;
            end
            % Set default values if not specified otherwise
            if ~isfield(obj.Params, 'learn_tempo_ranges')
                obj.Params.learn_tempo_ranges = 1;
            end
            if ~isfield(obj.Params, 'pattern_size')
                obj.Params.pattern_size = 'bar';
            end
            if ~isfield(obj.Params, 'min_tempo_bpm')
                obj.Params.min_tempo_bpm = 60;
            end
            if ~isfield(obj.Params, 'max_tempo_bpm')
                obj.Params.max_tempo_bpm = 220;
            end
            if ~isfield(obj.Params, 'same_tempo_per_meter')
                obj.Params.same_tempo_per_meter = 0;
            end
            if ~isfield(obj.Params, 'frame_length')
                obj.Params.frame_length = 0.02;
            end
            if ~isfield(obj.Params, 'whole_note_div')
                obj.Params.whole_note_div = 64;
            end
            if ~isfield(obj.Params, 'feat_type')
                obj.Params.feat_type = {'lo230_superflux.mvavg', ...
                    'hi250_superflux.mvavg'};
            end
            if ~isfield(obj.Params, 'observationModelType')
                obj.Params.observationModelType = 'MOG';
            end
            if ~isfield(obj.Params, 'save_beats')
                obj.Params.save_beats = 1;
            end
            if ~isfield(obj.Params, 'save_downbeats')
                obj.Params.save_downbeats = 0;
            end
            if ~isfield(obj.Params, 'save_median_tempo')
                obj.Params.save_median_tempo = 0;
            end
            if ~isfield(obj.Params, 'save_tempo_sequence')
                obj.Params.save_tempo_sequence = 0;
            end
            if ~isfield(obj.Params, 'save_rhythm')
                obj.Params.save_rhythm = 0;
            end
            if ~isfield(obj.Params, 'save_meter')
                obj.Params.save_meter = 0;
            end
            if ~isfield(obj.Params, 'save_features_to_file')
                obj.Params.save_features_to_file = 0;
            end
            if ~isfield(obj.Params, 'load_features_from_file')
                obj.Params.load_features_from_file = 1;
            end
            if ~isfield(obj.Params, 'transition_model_type')
                obj.Params.transition_model_type = '2015';
            end
            if strfind(obj.Params.inferenceMethod, 'HMM') > 0
                if strcmp(obj.Params.transition_model_type, '2015')
                    if ~isfield(obj.Params, 'alpha')
                        obj.Params.transition_params.transition_lambda = ...
                            100;
                    else
                        obj.Params.transition_params.transition_lambda = ...
                            obj.Params.alpha;
                    end
                elseif strfind(obj.Params.transition_model_type, '2006') > 0
                    if ~isfield(obj.Params, 'pn')
                        obj.Params.transition_params.pn = 0.01;
                    else
                        obj.Params.transition_params.pn = obj.Params.pn;
                    end
                end
            elseif strcmp(obj.Params.inferenceMethod(1:2), 'PF')
                if isfield(obj.Params, 'tempo_std_per')
                    obj.Params.transition_params.tempo_std_per = ...
                        obj.Params.tempo_std_per;
                else
                    obj.Params.transition_params.tempo_std_per = 0.02; % 2%
                end
            end
            if ~isfield(obj.Params, 'pattern_size')
                obj.Params.pattern_size = 'bar';
            end
            if ~isfield(obj.Params, 'dist_cluster')
                obj.Params.dist_cluster = 'data';
            end
            if ~isfield(obj.Params, 'use_silence_state')
                obj.Params.use_silence_state = 0;
            end
            if obj.Params.use_silence_state
                obj.Params.transition_params.p2s = Params.p2s;
                obj.Params.transition_params.pfs = Params.pfs;
            end
            if ~isfield(obj.Params, 'tempo_outlier_percentile')
                obj.Params.tempo_outlier_percentile = 5;
            end
            if ~isfield(obj.Params, 'reorganize_bars_into_cluster')
                obj.Params.reorganize_bars_into_cluster = 0;
            end
            if ~isfield(obj.Params, 'clusterIdFln') && ...
                    ~isfield(obj.Params, 'cluster_type')
                obj.Params.cluster_type = 'meter';
            end
            if ~isfield(obj.Params, 'store_training_data')
                obj.Params.store_training_data = 0;
            end
            if ~isfield(obj.Params, 'load_training_data')
                obj.Params.load_training_data = 0;
            end
            if ~isfield(obj.Params, 'stored_train_data_fln') && ...
                    (obj.Params.load_training_data || ...
                    obj.Params.store_training_data)
                % generate name
                featStr = '';
                for iDim = 1:length(obj.Params.feat_type)
                    featType = strrep(obj.Params.feat_type{iDim}, ...
                        '.', '-');
                    featStr = [featStr, '_', featType];
                end
                if ~isfield(obj.Params, 'train_set')
                    if iscell(obj.Params.trainLab)
                        obj.Params.train_set = 'custom';
                    else
                        [~, obj.Params.train_set, ~] = ...
                            fileparts(obj.Params.trainLab);
                    end
                end
                obj.Params.stored_train_data_fln = fullfile(...
                    obj.Params.data_path, [obj.Params.train_set, '_', ...
                    obj.Params.pattern_size, featStr, '.mat']);
            end
            if ~isfield(obj.Params, 'results_path')
                obj.Params.results_path = '.';
            else
                if ~exist(obj.Params.results_path, 'dir')
                    system(['mkdir ', obj.Params.results_path]);
                end
            end
            
        end
    end
    
    methods(Static)
        
        function [] = save_beats(beats, save_fln)
            % save beats and downbeats in the format
            % (beat time in [sec]) \tab (beat number)
            fid = fopen(save_fln, 'w');
            fprintf(fid, '%.3f\t%i\n', beats');
            fclose(fid);
        end
        
        function [] = save_downbeats(beats, save_fln)
            fid = fopen(save_fln, 'w');
            fprintf(fid, '%.3f\n', beats(beats(:, 2) == 1)');
            fclose(fid);
        end
        
        function [] = save_median_tempo(tempo, save_fln)
            fid = fopen(save_fln, 'w');
            fprintf(fid, '%i\n', median(tempo));
            fclose(fid);
        end
        
        function [] = save_tempo_sequence(tempo, ts, save_fln)
            dlmwrite(save_fln, [ts(:) tempo(:)], 'precision', '%.2f');
        end
        
        function [] = save_rhythm(rhythm, rhythm_names, ts, save_fln)
            rhythm_change_points = find(diff(rhythm(:)));
            rhythm_change_points = [1; rhythm_change_points; length(rhythm)];
            fid = fopen(save_fln, 'w');
            for i=1:length(rhythm_change_points)-1
                fprintf(fid, '%3.2f\t%3.2f\t%s\n', ts(rhythm_change_points(i)), ...
                    ts(rhythm_change_points(i+1)), ...
                    rhythm_names{rhythm(rhythm_change_points(i)+1)});
            end
            fclose(fid);
        end
        
        
        function [] = save_meter(meter, save_fln)
            m = unique(meter', 'rows')';
            fid = fopen(save_fln, 'w');
            fprintf(fid, '%i/%i\n', m(1), m(2));
            fclose(fid);
        end
        
    end
    
end
