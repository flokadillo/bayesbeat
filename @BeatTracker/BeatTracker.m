classdef BeatTracker < handle
    % Beat tracker Class
    properties
        input_fln               % input filename (.wav or feature file)
        model                   % probabilistic model
        inferenceMethod         % forward, viterbi, ...
        feature
        train_data
        test_data
        sim_dir                 % directory where results are saved
        temp_path
        init_model_fln          % fln of initial model to start with
        Params                  % parameters from confi file
    end
    
    methods
        function obj = BeatTracker(Params, sim_id)
            
            % parse probabilistic model
            if isfield(Params, 'model_fln') && ~isempty(Params.model_fln)
                if exist(Params.model_fln, 'file')
                    c = load(Params.model_fln);
                    fields = fieldnames(c);
                    obj.model = c.(fields{1});
                    obj.init_model_fln = Params.model_fln;
                    obj.feature = Feature(obj.model.obs_model.feat_type, obj.model.frame_length);
                else
                    error('Model file %s not found', model_fln);
                end
            else
                obj.feature = Feature(Params.feat_type, Params.frame_length);
            end
            obj.inferenceMethod = Params.inferenceMethod;
            if exist('sim_id', 'var')
                obj.sim_dir = fullfile(Params.results_path, num2str(sim_id));
            end
            if isfield(Params, 'temp_path')
                obj.temp_path = Params.temp_path;
            end
            obj.Params = Params;
        end
        
        function init_model(obj)
            if isfield(obj.Params, 'model_fln') && ~isempty(obj.Params.model_fln)
                if exist(obj.Params.model_fln, 'file')
                    c = load(obj.Params.model_fln);
                    fields = fieldnames(c);
                    obj.model = c.(fields{1});
                    obj.init_model_fln = obj.Params.model_fln;
                    fprintf('    Loaded model from %s\n', obj.Params.model_fln);
                else
                    fprintf('%s was not found, creating new model ...\n', obj.Params.model_fln);
                end
            else
                if isempty(obj.train_data) % no training data given -> set defaults
                    obj.train_data.rhythm2meter_state = ones(1, obj.Params.R); 
                    obj.train_data.rhythm_names = cellfun(@(x) num2str(x), num2cell(1:obj.Params.R), 'UniformOutput', false);
                end
                
                switch obj.Params.inferenceMethod(1:2)
                    case 'HM'
                        obj.model = HMM(obj.Params, obj.train_data.meter_state2meter, obj.train_data.rhythm2meter_state, obj.train_data.rhythm_names);
                    case 'PF'
                        obj.model = PF(obj.Params, obj.train_data.rhythm2meter_state, obj.train_data.rhythm_names);
                    otherwise
                        error('BeatTracker.init_model: inference method %s not known', obj.Params.inferenceMethod);
                end
            end
            
        end

        function init_train_data(obj)
            fprintf('* Set up training data ...');
            obj.train_data = Data(obj.Params.trainLab, 1);
            if ~isfield(obj.Params, 'clusterIdFln'), return;  end
            obj.train_data = obj.train_data.read_pattern_bars(obj.Params.clusterIdFln, obj.Params.pattern_size);
            % make filename of features
            [~, clusterFName, ~] = fileparts(obj.Params.clusterIdFln);
            featStr = '';
            for iDim = 1:obj.Params.featureDim
                featType = strrep(obj.Params.feat_type{iDim}, '.', '-');
                featStr = [featStr, featType];
            end
            featuresFln = fullfile(obj.Params.data_path, [clusterFName, '_', featStr, '.mat']);
            barGrid_eff = obj.Params.whole_note_div * (obj.train_data.meter_state2meter(1, :) ./ obj.train_data.meter_state2meter(2, :)); 
            obj.train_data = obj.train_data.extract_feats_per_file_pattern_barPos_dim(obj.Params.whole_note_div, ...
                barGrid_eff, obj.Params.featureDim, featuresFln, obj.Params.feat_type, ...
                obj.Params.frame_length, obj.Params.reorganize_bars_into_cluster);
            % process silence data
            if obj.Params.use_silence_state
                if length(obj.Params.feat_type) > 1
                    error('\nERROR: BeatTracker.init_train_data: So far, only one feature dimension supported\n', obj.Params.feat_type{1});
                end
                fid = fopen(obj.Params.silence_lab, 'r');
                silence_files = textscan(fid, '%s\n'); silence_files = silence_files{1};
                fclose(fid);
                obj.train_data.feats_silence = [];
                for iFile=1:length(silence_files)
                    obj.train_data.feats_silence = [obj.train_data.feats_silence;  obj.feature.load_feature(silence_files{iFile})];
                end
            end
            fprintf(' done\n');
        end
        
        function init_test_data(obj)
            % create test_data object
            obj.test_data = Data(obj.Params.testLab, 0);
            if isfield(obj.Params, 'test_annots_folder')
                obj.test_data = obj.test_data.set_annots_path(obj.Params.test_annots_folder);
            end
            % in case where test and train data are the same, cluster ids for the test
            % set are known and can be evaluated
%             if strcmp(obj.Params.train_set, obj.Params.test_set) && isfield(obj.Params, 'clusterIdFln')
%                 obj.test_data = obj.test_data.read_pattern_bars(obj.Params.clusterIdFln, obj.Params.meters, obj.Params.pattern_size);
%             end
        end
        
        function train_model(obj)
            if isempty(obj.init_model_fln)
                if isfield(obj.Params, 'min_tempo')
                    tempo_min_per_cluster = repmat(obj.Params.min_tempo, 1, obj.Params.R);
                    tempo_max_per_cluster = repmat(obj.Params.max_tempo, 1, obj.Params.R);
                else
                    [tempo_min_per_cluster, tempo_max_per_cluster] = obj.train_data.get_tempo_per_cluster();
                end
               
                obj.model = obj.model.make_transition_model(floor(min(tempo_min_per_cluster, [], 1)), ceil(max(tempo_max_per_cluster, [], 1)));
                
                if obj.Params.use_silence_state
                    obj.model = obj.model.make_observation_model(obj.train_data);
                else
                    obj.model = obj.model.make_observation_model(obj.train_data);
                end
                               
                obj.model = obj.model.make_initial_distribution([tempo_min_per_cluster; tempo_max_per_cluster]);
                
                fln = fullfile(obj.temp_path, 'last_model.mat');
                switch obj.inferenceMethod(1:2)
                    case 'HM'
                        hmm = obj.model;
                        save(fln, 'hmm');
                    case 'PF'
                        pf = obj.model;
                        save(fln, 'pf');
                end
                fprintf('* Saved model (Matlab) to %s\n', fln);
            end
            
%             obj.model.save_hmm_data_to_hdf5('~/diss/src/matlab/beat_tracking/bayes_beat/data/filip/');

            if isfield(obj.Params, 'viterbi_learning_iterations') && obj.Params.viterbi_learning_iterations > 0
            %    obj.model.trans_model = TransitionModel(obj.model.M, obj.model.Meff, obj.model.N, obj.model.R, obj.model.pn, obj.model.pr, ...
             %       obj.model.rhythm2meter_state, ones(1, obj.model.R), ones(1, obj.model.R)*obj.model.N);
                obj.refine_model(obj.Params.viterbi_learning_iterations);
            end
        end
        
        function train_transition_model(obj, tempo_per_cluster)
                obj.model = obj.model.make_transition_model(floor(min(tempo_per_cluster)), ceil(max(tempo_per_cluster)));
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
            %             fprintf('* Load features');
            %             observations = obj.feature.load_all_features(obj.train_data.file_list);
            %             fprintf(' ... done\n');
            if ~isempty(obj.init_model_fln) && ~isempty(strfind(obj.init_model_fln, '-'))
                dash = strfind(obj.init_model_fln, '-');
                iter_start = str2double(obj.init_model_fln(dash(end)+1:end-4)) + 1;
            else
%                 hmm = obj.model;
%                 save(fullfile(obj.sim_dir, ['hmm-', obj.train_data.dataset, '-0.mat']), 'hmm');
                iter_start = 1;
            end
            
            % read annotations and add them to train_data object:
            obj.train_data.read_beats;
            obj.train_data.read_meter;
            
            for i = iter_start:iter_start+iterations-1
                fprintf('* Viterbi training: iteration %i\n', i);
                [obj.model, bar2cluster] = obj.model.viterbi_training(obj.feature, obj.train_data);
                hmm = obj.model;
                save(fullfile(obj.sim_dir, ['hmm-', obj.train_data.dataset, '-', num2str(i), '.mat']), 'hmm');
                save(fullfile(obj.sim_dir, ['bar2cluster-', obj.train_data.dataset, '-', num2str(i), '.mat']), 'bar2cluster');
            end
        end
               
        function results = do_inference(obj, test_file_id, do_output)
            if ~exist('do_output', 'var')
                do_output = 1;
            end
            [~, fname, ~] = fileparts(obj.test_data.file_list{test_file_id});
            % load feature
            observations = obj.feature.load_feature(obj.test_data.file_list{test_file_id}, obj.Params.save_features_to_file);
            % compute observation likelihoods
            tic;
            results = obj.model.do_inference(observations, fname, obj.inferenceMethod, do_output);
            if do_output
                fprintf('    Real time factor: %.2f\n', toc / (size(observations, 1) * obj.feature.frame_length));
            end

            
%                         % save state sequence of annotations to file
%                         annot_fln = strrep(obj.test_data.file_list{test_file_id}, 'wav', 'beats');
%                         if exist(annot_fln, 'file')
%                             annots = load(annot_fln);
%                             r = obj.test_data.bar2cluster(find(obj.test_data.bar2file == test_file_id, 1));
%                             if isempty(r)
%                                 fprintf('    Cannot compute true path, file not in test_data included ...\n');
%                             else
%                                 [m, n] = HMM.getpath(obj.model.Meff(obj.model.rhythm2meter_state(r)), annots, obj.model.frame_length, size(observations, 1));
%                                 anns = [m, n, ones(length(m), 1) * r];
%                                 save(['~/diss/src/matlab/beat_tracking/bayes_beat/temp/', fname, '_anns.mat'], 'anns');
%                             end
%                         end
            %
            
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
            BeatTracker.save_beats(results{1}, fullfile(save_dir, [fname, '.beats.txt']));
            BeatTracker.save_tempo(results{2}, fullfile(save_dir, [fname, '.bpm']));
            BeatTracker.save_meter(results{3}, fullfile(save_dir, [fname, '.meter']));
            BeatTracker.save_rhythm(results{4}, fullfile(save_dir, [fname, '.rhythm']), ...
                obj.model.rhythm_names);
            if strfind(obj.inferenceMethod, 'HMM')
                BeatTracker.save_best_path(results{5}, fullfile(save_dir, [fname, '.best_path']));
            end
        end
    end
    
    methods(Static)
        function [] = save_beats(beats, save_fln)
            fid = fopen(save_fln, 'w');
            fprintf(fid, '%.3f\t%i.%i\n', beats');
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
        
        function [] = save_best_path(best_path, save_fln)
            fid = fopen(save_fln, 'w');
            fprintf(fid, '%i\n', best_path);
            fclose(fid);
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
