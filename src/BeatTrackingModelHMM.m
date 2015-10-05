classdef BeatTrackingModelHMM < handle & BeatTrackingModel
    % BeatTrackingModelHMM class
    % Subclass of BeatTrackingModel which implements HMM specific
    % functionality on how to set up a BeatTrackingHMM
    
    properties
        HMM
        initial_prob
        tm_type
        online_params
        obs_lik_floor
        use_mex_viterbi
    end
    
    methods
        function obj = BeatTrackingModelHMM(Params, Clustering)
            % Call superclass constructor
            obj@BeatTrackingModel(Params, Clustering);
            [State_space_params, store_proximity] = obj.parse_params(Params);
            if State_space_params.use_silence_state
                Clustering.rhythm_names{State_space_params.n_patterns + 1} = ...
                    'silence';
            end
            bar_durations = Clustering.rhythm2meter(:, 1) ./ ...
                Clustering.rhythm2meter(:, 2);
            % Create state_space
            if strcmp(obj.tm_type, '2015')
                State_space_params.max_positions = bar_durations;
                obj.state_space = BeatTrackingStateSpaceHMM2015(...
                    State_space_params, Params.min_tempo_bpm, ...
                    Params.max_tempo_bpm, Clustering.rhythm2nbeats, ...
                    Clustering.rhythm2meter, Params.frame_length, ...
                    Clustering.rhythm_names, ...
                    State_space_params.use_silence_state, ...
                    store_proximity);
            elseif strcmp(obj.tm_type, '2006')
                State_space_params.max_positions = ...
                    round(State_space_params.max_positions * bar_durations / ...
                    max(bar_durations));
                obj.state_space = BeatTrackingStateSpaceHMM2006(...
                    State_space_params, Params.min_tempo_bpm, ...
                    Params.max_tempo_bpm, Clustering.rhythm2nbeats, ...
                    Clustering.rhythm2meter, Params.frame_length, ...
                    Clustering.rhythm_names, ...
                    State_space_params.use_silence_state, ...
                    store_proximity);
            end
        end
        
        function train_model(obj, transition_probability_params, train_data, ...
                cells_per_whole_note, dist_type, results_path)
            obj.make_initial_distribution(train_data.meters);
            obj.make_transition_model(transition_probability_params);
            obj.make_observation_model(train_data, cells_per_whole_note, ...
                dist_type);
            obj.HMM = HiddenMarkovModel(obj.trans_model, obj.obs_model, ...
                obj.initial_prob);
            fln = fullfile(results_path, 'model.mat');
            hmm = obj;
            save(fln, 'hmm');
            fprintf('* Saved model (Matlab) to %s\n', fln);
        end
        
        function make_transition_model(obj, transition_probability_params)
            if strcmp(obj.tm_type, '2015')
                obj.trans_model = BeatTrackingTransitionModelHMM2015(...
                    obj.state_space, transition_probability_params);
            elseif strcmp(obj.tm_type, '2006')
                obj.trans_model = BeatTrackingTransitionModelHMM2006(...
                    obj.state_space, transition_probability_params);
            end
            % Check transition model
            if obj.trans_model.is_corrupt();
                error('Corrupt transition model');
            end
        end
        
        function make_observation_model(obj, train_data, ...
                cells_per_whole_note, dist_type)
            obj.obs_model = BeatTrackingObservationModelHMM(obj.state_space, ...
                train_data.feature.feat_type, dist_type, ...
                cells_per_whole_note);
            % Train model
            if ~strcmp(dist_type, 'RNN')
                obj.obs_model = obj.obs_model.train_model(train_data);
                obj.train_dataset = train_data.dataset;
            end
        end
        
        function make_initial_distribution(obj, meters)
            if ~exist('meters', 'var')
                obj.use_meter_prior = 0;
            end
            fprintf('* Set up initial distribution\n');
            n_states = obj.state_space.n_states;
            if obj.state_space.use_silence_state
                % always start in the silence state
                obj.initial_prob = zeros(n_states, 1);
                obj.initial_prob(end) = 1;
            else
                obj.initial_prob = ones(n_states, 1) / n_states;
            end
            if obj.use_meter_prior
                [unique_meters, ~, idx] = unique(meters, 'rows');
                meter_frequency = hist(idx, max(idx));
                meter_frequency = meter_frequency / sum(meter_frequency);
                
                for r = 1:obj.state_space.n_patterns
                    idx_r = (obj.state_space.pattern_from_state == r);
                    idx_m = ismember(unique_meters, ...
                        obj.state_space.meter_from_pattern(r, :), 'rows');
                    obj.initial_prob(idx_r) = obj.initial_prob(idx_r) * ...
                        meter_frequency(idx_m);
                    fprintf('    Weighting meter %i/%i by %.2f\n', ...
                        unique_meters(idx_m, 1), ...
                        unique_meters(idx_m, 2), ...
                        meter_frequency(idx_m));
                end
                obj.initial_prob = obj.initial_prob / sum(obj.initial_prob);
            end
        end
        
        function results = do_inference(obj, y, inference_method, fname, ...
                belief_func)
            if obj.hmm_is_corrupt
                error('    WARNING: @HMM/do_inference.m: HMM is corrupt\n');
            end
            if strcmp(inference_method, 'HMM_forward')
                [~, ~, hidden_state_sequence, ~] = obj.forward_path(obs_lik, ...
                    do_output, fname, y);
            elseif strcmp(inference_method, 'HMM_viterbi')
                % decode MAP state sequence using Viterbi
                fprintf('* Decoding (viterbi) .');
                if exist('belief_func', 'var')
                    % use viterbi with belief functions
                    hidden_state_sequence = obj.viterbi_iteration(obs_lik, ...
                        belief_func);
                else
                    if obj.use_mex_viterbi
                        hidden_state_sequence = obj.HMM.viterbi(y, 0);
                        fprintf('\n');
                    else
                        hidden_state_sequence = obj.HMM.viterbi(y, ...
                            obj.use_mex_viterbi);
                    end
                end
            else
                error('inference method not specified\n');
            end
            % decode state index into sub indices
            m_path = obj.state_space.position_from_state(...
                hidden_state_sequence)';
            n_path = obj.state_space.tempo_from_state(...
                hidden_state_sequence)';
            r_path = obj.state_space.pattern_from_state(...
                hidden_state_sequence)';
            results = obj.convert_state_sequence(m_path, n_path, r_path, y);
        end
        
        function belief_func = make_belief_function(obj, Constraint)
            if ismember('downbeats', Constraint.type)
                c = find(ismember(Constraint.type, 'downbeats'));
                tol_downbeats = 0.05; % given in beat proportions
                % compute tol_win in [frames]
                % belief_func:
                % col1: frames where annotation is available,
                % col2: sparse vector that is one for possible states
                belief_func = cell(length(Constraint.data{c}), 2);
                beatpositions = obj.beat_positions;
                bar_pos_per_beat = obj.state_space.max_position_from_pattern ./ ...
                    obj.state_space.n_beats_from_pattern;
                win_pos = tol_downbeats .* bar_pos_per_beat;
                for i_db = 1:length(Constraint.data{c})
                    i_frame = max([1, round(Constraint.data{c}(i_db) / ...
                        obj.frame_length)]);
                    belief_func{i_db, 1} = i_frame;
                    idx = false(obj.trans_model.num_states, 1);
                    % how many bar positions are one beat?
                    for i_r = 1:obj.state_space.n_patterns
                        idx_r = obj.trans_model.mapping_state_rhythm == ...
                            i_r;
                        win_left = obj.trans_model.mapping_state_position > ...
                            (beatpositions{i_r}(end) +...
                            bar_pos_per_beat(i_r) - win_pos(i_r));
                        win_right = obj.trans_model.mapping_state_position < ...
                            (beatpositions{i_r}(1) + win_pos(i_r));
                        idx = idx | (idx_r & (win_left | win_right));
                    end
                    belief_func{i_db, 2} = idx;
                end
            end
            if ismember('beats', Constraint.type)
                c = find(ismember(Constraint.type, 'beats'));
                tol_beats = 0.1; % given in beat proportions
                tol_tempo = 0.4; % given in percent of the actual tempo
                n_beats = length(Constraint.data{c});
                beatpositions = obj.beat_positions;
                % find states which are considered in the window
                idx = false(obj.trans_model.num_states, 1);
                state_pos_per_beat = obj.state_space.max_position_from_pattern ./ ...
                    obj.state_space.n_beats_from_pattern;
                for i_r = 1:obj.state_space.n_patterns
                    win_pos = tol_beats * state_pos_per_beat(i_r);
                    for i_b = 1:length(beatpositions{i_r})
                        win_left = obj.trans_model.mapping_state_position > ...
                            (beatpositions{i_r}(i_b) - win_pos);
                        win_right = obj.trans_model.mapping_state_position < ...
                            (beatpositions{i_r}(i_b) + win_pos);
                        idx = idx | (win_left & win_right);
                    end
                end
                belief_func = cell(n_beats, 2);
                ibi = diff(Constraint.data{c});
                for i_db = 1:n_beats
                    idx_b = false(obj.trans_model.num_states, 1);
                    ibi_i = mean(ibi(max([1, i_db-1]):...
                        min([n_beats-1, i_db])));
                    % loop through rhythms, because each tempo (ibi_b)
                    % in BPM maps to a different state-space tempo
                    % which depends on whether we have eight note beats or
                    % quarter note beats
                    for i_r = 1:obj.state_space.n_patterns
                        idx_r = (obj.trans_model.mapping_state_rhythm ...
                            == i_r);
                        tempo_ss = state_pos_per_beat(i_r) * ...
                            obj.frame_length / ibi_i;
                        idx_low = obj.trans_model.mapping_state_tempo ...
                            > tempo_ss * (1 - tol_tempo);
                        idx_hi = obj.trans_model.mapping_state_tempo ...
                            < tempo_ss * (1 + tol_tempo);
                        idx_b = idx_b | (idx_r & idx_low & idx_hi);
                    end
                    i_frame = max([1, round(Constraint.data{c}(i_db) / ...
                        obj.frame_length)]);
                    belief_func{i_db, 1} = i_frame;
                    belief_func{i_db, 2} = idx_b & idx;
                end
            end
            if ismember('meter', Constraint.type)
                c = find(ismember(Constraint.type, 'meter'));
                valid_rhythms = find(ismember(obj.rhythm2meter, ...
                    Constraint.data{c}, 'rows'));
                idx = false(obj.trans_model.num_states, 1);
                for i_r=valid_rhythms(:)'
                    idx(obj.trans_model.mapping_state_rhythm == i_r) = ...
                        true;
                end
                % loop through existing belief functions
                % TODO: what if there are none?
                for b=1:size(belief_func, 1)
                    belief_func{b, 2} = belief_func{b, 2} & idx;
                    if sum(belief_func{b, 2}) == 0;
                        error('Belief function error\n');
                    end
                end
            end
        end
        
    end
    
    methods (Access = protected)
        
        function hmm_corrupt = hmm_is_corrupt(obj)
            num_states_hypothesis = [length(obj.initial_prob);
                length(obj.obs_model.cell_from_state);
                size(obj.trans_model.A, 1);
                length(obj.state_space.position_from_state);
                length(obj.state_space.tempo_from_state);
                length(obj.state_space.pattern_from_state)];
            % remove zeros which come from older model versions
            num_states_hypothesis = ...
                num_states_hypothesis(num_states_hypothesis > 0);
            if any(diff(num_states_hypothesis))
                hmm_corrupt = true;
                num_states_hypothesis
            else
                hmm_corrupt = false;
            end
        end
        
        
        function [State_space_params, store_proximity] = ...
                parse_params(obj, Params)
            % Store parameters and set defaults
            if isfield(Params, 'transition_model_type')
                obj.tm_type = Params.transition_model_type;
            else
                obj.tm_type = '2015';
            end
            if isfield(Params, 'online')
                obj.online_params.max_shift = Params.online.online_params.max_shift;
                obj.online_params.update_interval = Params.online.online_params.update_interval;
                obj.obs_lik_floor = Params.online.obs_lik_floor;
                store_proximity = 1;
            else
                store_proximity = 0;
            end
            if isfield(Params, 'use_mex_viterbi')
                obj.use_mex_viterbi = Params.use_mex_viterbi;
            else
                obj.use_mex_viterbi = 1;
            end
            if isfield(Params, 'N')
                State_space_params.n_tempi = Params.N;
            elseif strcmp(obj.tm_type, '2015')
                State_space_params.n_tempi = nan;
            elseif strcmp(obj.tm_type, '2006')
                State_space_params.n_tempi = 30;
            end
            if strcmp(obj.tm_type, '2006')
                if isfield(Params, 'M')
                    State_space_params.max_positions = Params.M;
                else
                    State_space_params.max_positions = 1600;
                end
            end
            State_space_params.n_patterns = Params.R;
            State_space_params.use_silence_state = Params.use_silence_state;
        end
    end    
end

