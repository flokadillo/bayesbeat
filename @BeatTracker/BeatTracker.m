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
            % save parameters
            obj.Params = Params;
            % parse parameters and set defaults
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
            if ~isfield(obj.Params, 'min_tempo')
                obj.Params.min_tempo = 60;
            end
            if ~isfield(obj.Params, 'max_tempo')
                obj.Params.max_tempo = 220;
            end
            if ~isfield(obj.Params, 'frame_length')
                obj.Params.frame_length = 0.02;
            end
            if ~isfield(obj.Params, 'whole_note_div')
                obj.Params.whole_note_div = 64;
            end
            if ~isempty(strfind(obj.Params.inferenceMethod, 'HMM')) && ...
                    ~isfield(obj.Params, 'pn')
                obj.Params.pn = 0.01;  
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
            if ~isfield(obj.Params, 'save_tempo')
                obj.Params.save_tempo = 0;
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
            if ~isfield(obj.Params, 'alpha')
                obj.Params.alpha = 100;
            end
            if ~isfield(obj.Params, 'pattern_size')
                obj.Params.pattern_size = 'bar';
            end
            if ~isfield(obj.Params, 'use_silence_state')
                obj.Params.use_silence_state = 0;
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
                obj.Params.stored_train_data_fln = fullfile(...
                    obj.Params.data_path, [obj.Params.train_set, '_', ...
                    obj.Params.pattern_size, featStr, '.mat']);
            end
            % load or create probabilistic model
            obj.init_model();            
        end
        
        
        function init_model(obj)
            if isfield(obj.Params, 'model_fln') && ~isempty(obj.Params.model_fln)
                if exist(obj.Params.model_fln, 'file')
                    c = load(obj.Params.model_fln);
                    fields = fieldnames(c);
                    obj.model = c.(fields{1});
                    obj.model = obj.convert_to_new_model_format(obj.model);
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
                if strcmp(obj.Params.observationModelType, 'RNN')
                    obj.train_data.clustering.rhythm2nbeats = 1;
                    obj.train_data.clustering.rhythm2meter = [1, 4];
                    obj.train_data.clustering.rhythm_names = {'rnn'};
                    obj.train_data.clustering.pr = 1;
                    obj.train_data.feature = obj.feature;
                    obj.Params.R = 1;
                else
                    % initialise training data to see how many pattern states
                    % we need and which time signatures have to be modelled
                    obj.init_train_data();
                end
                    switch obj.Params.inferenceMethod(1:2)
                        case 'HM'
                            obj.model = HMM(obj.Params, ...
                                obj.train_data.clustering);
                        case 'PF'
                            obj.model = PF(obj.Params, ...
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
                obj.train_data.organise_feats_into_bars(...
                    obj.Params.whole_note_div);
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
                % Check if cluster assignment file is given. If yes load
                % cluster assignments from yes, if no compute them.
                if isfield(obj.Params, 'clusterIdFln') && ...
                        exist(obj.Params.clusterIdFln, 'file')
                    obj.train_data = obj.train_data.read_pattern_bars(...
                        obj.Params.clusterIdFln, obj.Params.pattern_size);
                    if ~isempty(obj.train_data.pr)
                        % if pr is given in cluster file save it here
                        obj.Params.pr = obj.train_data.pr;
                    end
                else
                    fprintf('* Clustering data by %s ...', ...
                        obj.Params.cluster_type);
                    if ismember(obj.Params.cluster_type, {'meter', ...
                            'rhythm'})
                        obj.train_data.cluster_from_labels(...
                            obj.Params.cluster_type);
                    elseif strcmp(obj.Params.cluster_type, 'kmeans')
                        obj.train_data.cluster_from_features(...
                            obj.Params.n_clusters);
                    end
                    fprintf('done\n  %i clusters detected.\n', ...
                        obj.train_data.clustering.n_clusters);
                end
                % save extracted training data
                if obj.Params.store_training_data
                    data = obj.train_data;
                    save(obj.Params.stored_train_data_fln, 'data');
                end
            end
            obj.Params.R = obj.train_data.clustering.n_clusters;
        end
        
        function init_test_data(obj)
            % create test_data object
            obj.test_data = Data(obj.Params.testLab, obj.Params.feat_type, ...
                obj.Params.frame_length, obj.Params.pattern_size);
        end
        
        function train_model(obj)
            if isempty(obj.init_model_fln)
                if obj.Params.learn_tempo_ranges
                    % get tempo ranges from data for each file
                    [tempo_min_per_cluster, tempo_max_per_cluster] = ...
                        obj.train_data.get_tempo_per_cluster(...
                        obj.Params.tempo_outlier_percentile);
                    % find min/max for each pattern
                    tempo_min_per_cluster = min(tempo_min_per_cluster)';
                    tempo_max_per_cluster = max(tempo_max_per_cluster)';
                    % restrict ranges
                    tempo_min_per_cluster(tempo_min_per_cluster < ...
                        obj.Params.min_tempo) = obj.Params.min_tempo;
                    tempo_max_per_cluster(tempo_max_per_cluster > ...
                        obj.Params.max_tempo) = obj.Params.max_tempo;
                else
                    tempo_min_per_cluster = repmat(obj.Params.min_tempo, 1, ...
                        obj.Params.R);
                    tempo_max_per_cluster = repmat(obj.Params.max_tempo, 1, ...
                        obj.Params.R);
                end
                fprintf('* Set up transition model\n');
                obj = obj.train_transition_model(tempo_min_per_cluster, ...
                    tempo_max_per_cluster);
                fprintf('* Set up observation model\n');
                if obj.Params.use_silence_state
                    obj.model = obj.model.make_observation_model(...
                        obj.train_data);
                else
                    obj.model = obj.model.make_observation_model(...
                        obj.train_data);
                end
                fprintf('* Set up initial distribution\n');
                obj.model = obj.model.make_initial_distribution(...
                    [tempo_min_per_cluster; tempo_max_per_cluster]);
                % save trained model
                fln = fullfile(obj.Params.results_path, 'model.mat');
                switch obj.Params.inferenceMethod(1:2)
                    case 'HM'
                        hmm = obj.model;
                        save(fln, 'hmm');
                    case 'PF'
                        pf = obj.model;
                        save(fln, 'pf');
                end
                fprintf('* Saved model (Matlab) to %s\n', fln);
            end
            if isfield(obj.Params, 'viterbi_learning_iterations') && ...
                    obj.Params.viterbi_learning_iterations > 0
                obj.refine_model(obj.Params.viterbi_learning_iterations);
            end
        end
        
        
        function obj = train_transition_model(obj, tempo_min_per_cluster, ...
                tempo_max_per_cluster)
            switch obj.Params.inferenceMethod(1:2)
                case 'HM'
                    obj.model = obj.model.make_transition_model(...
                        tempo_min_per_cluster, tempo_max_per_cluster, ...
                        obj.Params.alpha, obj.Params.pn, ...
                        obj.train_data.clustering.pr);
                case 'PF'
                    obj.model = obj.model.make_transition_model(...
                        tempo_min_per_cluster, tempo_max_per_cluster, ...
                        obj.Params.alpha, obj.Params.sigmaN, ...
                        obj.train_data.clustering.pr);
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
        
        function test_file_ids = retrain_model(obj, test_files_to_exclude)
            % test_files_to_exclude can be either a scalar
            % (index of the file to be excluded for leave-one-out
            % validation) or a cell array of file names to be excluded.
            fprintf('* Retraining observation model ');
            file_idx = zeros(length(obj.train_data.file_list), 1);
            if length(test_files_to_exclude) == 1
                % leave one out: get pattern idx of test file to only
                % retrain the remaining patterns
                r_i = unique(obj.train_data.clustering.bar2cluster(...
                    obj.train_data.clustering.bar2file == ...
                    test_files_to_exclude));
                file_idx(test_files_to_exclude) = 1;
                test_file_ids = test_files_to_exclude;
            else
                % retrain all patterns
                r_i = 1:obj.model.R;
                % Get file indices of test_files_to_exclude within the training set
                for test_id = 1:length(test_files_to_exclude)
                    for train_id = 1:length(obj.train_data.file_list)
                        if strfind(obj.train_data.file_list{train_id}, ...
                                test_files_to_exclude{test_id});
                            file_idx(train_id) = 1;
                            break;
                        end
                    end
                end
                test_file_ids = find(file_idx)';
            end
            % exclude test files from training:
            obj.model = ...
                obj.model.retrain_observation_model(...
                obj.train_data.feats_file_pattern_barPos_dim(~file_idx, :, ...
                :, :), r_i);
            fprintf('done\n');
        end
        
        
        function refine_model(obj, iterations)
            if ~isempty(obj.init_model_fln) && ~isempty(strfind(obj.init_model_fln, '-'))
                dash = strfind(obj.init_model_fln, '-');
                iter_start = str2double(obj.init_model_fln(dash(end)+1:end-4)) + 1;
            else
                iter_start = 1;
            end
            % read annotations and add them to train_data object:
            for i = iter_start:iter_start+iterations-1
                fprintf('* Viterbi training: iteration %i\n', i);
                [obj.model, ~] = obj.model.viterbi_training(obj.feature, obj.train_data);
                hmm = obj.model;
                save(fullfile(obj.Params.results_path, ['hmm-', obj.train_data.dataset, '-', num2str(i), '.mat']), 'hmm');
                save(fullfile(obj.Params.results_path, ['bar2cluster-', obj.train_data.dataset, '-', num2str(i), '.mat']), 'bar2cluster');
            end
        end
        
        
        function results = do_inference(obj, test_file_id, do_output)
            if ~exist('do_output', 'var')
                do_output = 1;
            end
            [~, fname, ~] = fileparts(obj.test_data.file_list{test_file_id});
            % load feature
            observations = obj.feature.load_feature(...
                obj.test_data.file_list{test_file_id}, ...
                obj.Params.save_features_to_file, ...
                obj.Params.load_features_from_file);
            if isfield(obj.Params, 'constraint_type')
                Constraint.type = obj.Params.constraint_type;
                Constraint.data = obj.load_constraints(test_file_id);
                belief_func = obj.model.make_belief_function(Constraint);
                results = obj.model.do_inference(observations, fname, ...
                    obj.Params.inferenceMethod, belief_func);
            else
                results = obj.model.do_inference(observations, fname, ...
                    obj.Params.inferenceMethod);
            end
            % compute observation likelihoods
            
            if obj.model.save_inference_data
                % save state sequence of annotations to file
                annot_fln = strrep(strrep(obj.test_data.file_list{test_file_id}, ...
                    'wav', 'beats'), 'audio', 'annotations/beats');
                if exist(annot_fln, 'file')
                    annots = load(annot_fln);
                    meter = max(annots(:, 2));
                    r = find(obj.model.rhythm2meter(:, 1) == meter);
                    if isempty(r)
                        fprintf(['    Cannot compute true path, file not', ...
                            'in test_data included ...\n']);
                    else
                        [m, n] = HMM.getpath(obj.model.Meff(r), annots, ...
                            obj.model.frame_length, size(observations, 1));
                        anns = [m, n, ones(length(m), 1) * r];
                        save(['/tmp/', fname, '_anns.mat'], 'anns');
                    end
                end
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
            if obj.Params.save_tempo
                BeatTracker.save_tempo(results{2}, fullfile(save_dir, ...
                    [fname, '.bpm.txt']));
            end
            if obj.Params.save_meter
                BeatTracker.save_meter(results{3}, fullfile(save_dir, ...
                    [fname, '.meter.txt']));
            end
            if obj.Params.save_rhythm
                BeatTracker.save_rhythm(results{4}, fullfile(save_dir, ...
                    [fname, '.rhythm.txt']), obj.model.rhythm_names);
            end
        end
    end
    
    methods(Static)
        function [] = save_beats(beats, save_fln)
            fid = fopen(save_fln, 'w');
            fprintf(fid, '%.3f\t%i\n', beats');
            fclose(fid);
        end
        
        function [] = save_downbeats(beats, save_fln)
            fid = fopen(save_fln, 'w');
            fprintf(fid, '%.3f\n', beats(beats(:, 2) == 1)');
            fclose(fid);
        end
        
        function [] = save_tempo(tempo, save_fln)
            fid = fopen(save_fln, 'w');
            fprintf(fid, '%i\n', median(tempo));
            fclose(fid);
        end
        
        function [] = save_rhythm(rhythm, save_fln, rhythm_names)
            r = unique(rhythm);
            fid = fopen(save_fln, 'w');
            for i=1:length(r)
                fprintf(fid, '%s\n', rhythm_names{r(i)});
            end
            fclose(fid);
        end
        
        
        function [] = save_meter(meter, save_fln)
            m = unique(meter', 'rows')';
            fid = fopen(save_fln, 'w');
            fprintf(fid, '%i/%i\n', m(1), m(2));
            fclose(fid);
        end
        
                
        function new_model = convert_to_new_model_format(old_model)
            new_model = old_model.convert_old_model_to_new();
        end
        
    end
    
end
