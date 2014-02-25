classdef BeatTracker < handle
    % Beat tracker Class
    properties (SetAccess=private)
        input_fln               % input filename (.wav or feature file)
        model                   % probabilistic model
        inferenceMethod         % forward, viterbi, ...
        feature
        train_data
        test_data
        sim_dir                 % directory where results are save
        viterbi_learning_iterations
        Params                  % Parameters from config file
    end
    
    methods
        function obj = BeatTracker(Params, sim_id, model_fln)
            
            % parse probabilistic model
            if ~isempty(model_fln)
                if exist(model_fln, 'file')
                    c = load(model_fln);
                    fields = fieldnames(c);
                    obj.model = c.(fields{1});
                else
                    error('Model file %s not found', model_fln);
                end
            end
            obj.feature = Feature(Params.feat_type, Params.frame_length);
            obj.inferenceMethod = Params.inferenceMethod;
            obj.sim_dir = fullfile(Params.results_path, num2str(sim_id));
            obj.viterbi_learning_iterations = Params.viterbi_learning_iterations;
            obj.Params = Params;
            bar_durations = Params.meters(1, :) ./ Params.meters(2, :);
            meter2M = Params.M ./ max(bar_durations);
            obj.Params.Meff = round(bar_durations * meter2M);
            obj.Params.barGrid_eff = Params.whole_note_div * bar_durations; % number of grid points per meter
        end
        
        function init_model(obj)
            if isfield(obj.Params, 'model_fln')
                if exist(obj.Params.model_fln, 'file')
                    c = load(obj.Params.model_fln);
                    fields = fieldnames(c);
                    obj.model = c.(fields{1});
                end
            else
                
                
                switch obj.Params.inferenceMethod(1:2)
                    case 'HM'
                        obj.model = HMM(obj.Params, obj.train_data.rhythm2meter);
                    case 'PF'
                        obj.model = PF(obj.Params, obj.train_data.rhythm2meter);
                    otherwise
                        error('BeatTracker.init_model: inference method %s not known', obj.Params.inferenceMethod);
                end
            end
            
        end
        
        function init_train_data(obj)
            % create train_data object
            obj.train_data = Data(obj.Params.trainLab, 1, obj.feature.frame_length);
            %             obj.train_data = obj.train_data.set_annots_path(obj.Params.train_annots_folder);
            obj.train_data = obj.train_data.read_pattern_bars(obj.Params.clusterIdFln, obj.Params.meters, obj.Params.pattern_size);
            %             obj.train_data = obj.train_data.filter_out_meter([3, 4]);
            obj.train_data = obj.train_data.extract_feats_per_file_pattern_barPos_dim(obj.Params.whole_note_div, ...
                obj.Params.barGrid_eff, obj.Params.featureDim, obj.Params.featuresFln, obj.Params.feat_type, ...
                obj.Params.frame_length, obj.Params.reorganize_bars_into_cluster);
            if obj.Params.use_silence_state
                [dataPath, fname, featExt] = fileparts(obj.Params.silence_fln);
                feat_fln = fullfile(dataPath, 'beat_activations', [fname, '.', obj.Params.feat_type{1}]);
                obj.train_data = obj.train_data.learn_silence_state(feat_fln);
            end
        end
        
        function init_test_data(obj)
            % create test_data object
            obj.test_data = Data(obj.Params.testLab, 0, obj.feature.frame_length);
            if isfield(obj.Params, 'test_annots_folder')
                obj.test_data = obj.test_data.set_annots_path(obj.Params.test_annots_folder);
                obj.test_data = obj.test_data.filter_out_meter([3, 4]);
            end
            % in case where test and train data are the same, cluster ids for the test
            % set are known and can be evaluated
            if strcmp(obj.Params.train_set, obj.Params.test_set)
                obj.test_data = obj.test_data.read_pattern_bars(obj.Params.clusterIdFln, obj.Params.meters, obj.Params.pattern_size);
            end
        end
        
        function train_model(obj, use_tempo_prior)
            tempo_per_cluster = obj.train_data.get_tempo_per_cluster();
            if use_tempo_prior
                % define max/min tempo for each rhythm separately
                maxTempo = ceil(max(tempo_per_cluster));
                minTempo = floor(min(tempo_per_cluster));
            else
                % define the max/min tempo to be the same for all rhythms
                maxTempo = repmat(ceil(max(tempo_per_cluster(:))), 1, obj.model.R);
                minTempo = repmat(floor(min(tempo_per_cluster(:))), 1, obj.model.R);
            end
            obj.model = obj.model.make_transition_model(minTempo, maxTempo);
            
            if obj.Params.use_silence_state
                obj.model = obj.model.make_observation_model(obj.train_data.feats_file_pattern_barPos_dim, obj.train_data.feats_silence);
            else
                obj.model = obj.model.make_observation_model(obj.train_data.feats_file_pattern_barPos_dim);
            end
            
            obj.model = obj.model.make_initial_distribution(use_tempo_prior, tempo_per_cluster);
            
            obj.model.save_hmm_data_to_text('~/diss/src/matlab/beat_tracking/bayes_beat/data/filip/');
            
            if obj.viterbi_learning_iterations > 0
                obj.refine_model(obj.viterbi_learning_iterations);
            end
        end
        
        function retrain_model(obj, exclude_test_file_id)
            fprintf('    Retraining observation model ');
            if length(exclude_test_file_id) == 1
                r_i = unique(obj.train_data.bar2cluster(obj.train_data.bar2file == exclude_test_file_id));
            else
                r_i = 1:obj.model.R;
            end
            file_idx = ismember(1:length(obj.train_data.file_list), exclude_test_file_id);
            % exclude test files from training:
            obj.model = ...
                obj.model.retrain_observation_model(obj.train_data.feats_file_pattern_barPos_dim(~file_idx, :, :, :), r_i);
            fprintf('done\n');
        end
        
        function refine_model(obj, iterations)
            fprintf('* Set up belief functions');
            %             profile on
            belief_func = obj.train_data.make_belief_functions(obj.model);
            %             profile viewer
            fprintf(' ... done\n');
            fprintf('* Load features');
            observations = obj.feature.load_all_features(obj.train_data.file_list);
            fprintf(' ... done\n');
            hmm = obj.model;
            save(fullfile(obj.sim_dir, ['hmm-', obj.train_data.dataset, '-0.mat']), 'hmm');
            
            for i = 1:iterations
                fprintf('* Viterbi training: iteration %i\n', i);
                %                 profile on
                [obj.model, bar2cluster] = obj.model.viterbi_training(observations, belief_func, obj.train_data);
                %                 profile viewer
                hmm = obj.model;
                save(fullfile(obj.sim_dir, ['hmm-', obj.train_data.dataset, '-', num2str(i), '.mat']), 'hmm');
                save(fullfile(obj.sim_dir, ['bar2cluster-', obj.train_data.dataset, '-', num2str(i), '.mat']), 'bar2cluster');
            end
        end
        
        function load_features(obj, input_fln)
            obj.feature = obj.feature.load_feature(input_fln);
        end
        
        function compute_features(obj, input_fln)
            obj.input_fln = input_fln;
        end
        
        function results = do_inference(obj, test_file_id)
            [~, fname, ~] = fileparts(obj.test_data.file_list{test_file_id});
            % load feature
            obj.feature = obj.feature.load_feature(obj.test_data.file_list{test_file_id});
            % compute observation likelihoods
            [beats, tempo, rhythm, meter] = obj.model.do_inference(obj.feature.feature, fname);
            results{1} = beats;
            results{2} = tempo;
            results{3} = meter;
            results{4} = rhythm;
            annot_fln = strrep(obj.feature.input_fln, 'wav', 'beats');
            if exist(annot_fln, 'file')
                annots = load(annot_fln);
                r = obj.test_data.bar2cluster(find(obj.test_data.bar2file == test_file_id, 1));
                if isempty(r)
                    fprintf('    Cannot compute true path, file not in test_data included ...\n');
                else
                    [m, n] = HMM.getpath(obj.model.Meff(obj.model.rhythm2meter(r)), annots, obj.model.frame_length, size(obj.feature.feature, 1));
                    anns = [m, n, ones(length(m), 1) * r];
                    save(['~/diss/src/matlab/beat_tracking/bayes_beat/temp/', fname, '_anns.mat'], 'anns');
                end
            end
            %
            
        end
        
        function load_model(obj, fln)
            temp = load(fln);
            names = fieldnames(temp);
            obj.model = temp.(names{1});
        end
        
        function [] = save_results(obj, results, save_dir, fname)
            BeatTracker.save_beats(results{1}, fullfile(save_dir, [fname, '.beats']));
            BeatTracker.save_tempo(results{2}, fullfile(save_dir, [fname, '.bpm']));
            BeatTracker.save_meter(results{3}, fullfile(save_dir, [fname, '.meter']));
            BeatTracker.save_rhythm(results{4}, fullfile(save_dir, [fname, '.rhythm']), ...
                obj.train_data.rhythm_names);
        end
    end
    
    methods(Static)
        function [] = save_beats(beats, save_fln)
            fid = fopen(save_fln, 'w');
            fprintf(fid, '%.3f\t%.1f\n', beats');
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
            dlmwrite(save_fln, m, 'delimiter', '\t' );
        end
        
        function smoothedBeats = smooth_beats_sequence(inputBeats, win)
            % smooth_inputBeats(inputBeatFile, outputBeatFile, win)
            %   smooth beat sequence according to Dixon et al., Perceptual Smoothness
            %   of Tempo in Expressively Performed Music (2004)
            % ----------------------------------------------------------------------
            % INPUT Parameter:
            %   win                 :
            %
            % OUTPUT Parameter:
            %   Params            : structure array with beat tracking Paramseters
            %
            % 11.06.2012 by Florian Krebs
            % ----------------------------------------------------------------------
            
            if win < 1
                smoothedBeats = inputBeats;
                return
            end
            d = diff(inputBeats);
            
            % to correct for missing values at the ends, the sequence d is extended by
            % defining
            d = [d(1+win:-1:2); d; d(end-1:-1:end-win)];
            dSmooth = zeros(length(d), 1);
            
            for iBeat = 1+win:length(d)-win
                dSmooth(iBeat) = sum(d(iBeat-win:iBeat+win)) / (2*win+1);
            end
            dSmooth=dSmooth(win+1:end-win);
            % plot(d, 'r');
            
            smoothedBeats = inputBeats;
            smoothedBeats(2:end) = inputBeats(1) + cumsum(dSmooth);
            
        end
        
    end
    
end
