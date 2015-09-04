classdef BeatTrackerHMM < handle
    %UNTITLED2 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        state_space         % BeatTrackingStateSpace object
        HMM                 % HiddenMarkovModel object
        max_bar_cells       % number of observation distributions
        %                       of the longest bar (e.g., 64)
        frame_length        % audio frame length in [sec]
        dist_type           % type of parametric distribution
        trans_model         % transition model
        tm_type             % transition model type ('whiteley' or '2015')
        obs_model           % observation model
        initial_prob        % initial state distribution
        inferenceMethod
        train_dataset       % dataset, which HMM was trained on [string]
        correct_beats       % [0, 1, 2] correct beats after detection
        max_shift           % frame shifts that are allowed in forward path
        obs_lik_floor       % obs_lik has to be > floor to avoid overfitting
        update_interval     % best state sequence is set to the global max each 
        %                       update_interval
        use_silence_state
        use_mex_viterbi     % 1: use it, 0: don't use it (~5 times slower)
        beat_positions      % cell array; contains the bar positions of the 
        %                       beats for each rhythm
    end
    
    methods
        function obj = BeatTrackerHMM(Params, Clustering)
            [State_space_params, store_proximity] = obj.parse_params(Params);
            bar_durations = Clustering.rhythm2meter(:, 1) ./ ...
                Clustering.rhythm2meter(:, 2);
            obj.max_bar_cells = max(Params.whole_note_div * bar_durations);
            if obj.use_silence_state
                Clustering.rhythm_names{obj.state_space.n_patterns + 1} = ...
                    'silence';
            end
            % Create state_space
            if strcmp(obj.tm_type, '2015')
                State_space_params.max_positions = bar_durations;
                obj.state_space = BeatTrackingStateSpace2015(...
                    State_space_params, Params.min_tempo_bpm, ...
                    Params.max_tempo_bpm, Clustering.rhythm2nbeats, ...
                    Clustering.rhythm2meter, Params.frame_length, ...
                    Clustering.rhythm_names, obj.use_silence_state, ...
                    store_proximity);
            elseif strcmp(obj.tm_type, 'whiteley') % TODO: rename to 2006
                State_space_params.max_positions = ...
                    round(State_space_params.max_positions * bar_durations);
                obj.state_space = BeatTrackingStateSpace2006(...
                    State_space_params, Params.min_tempo_bpm, ...
                    Params.max_tempo_bpm, Clustering.rhythm2nbeats, ...
                    Clustering.rhythm2meter, Params.frame_length, ...
                    Clustering.rhythm_names, obj.use_silence_state, ...
                    store_proximity);
            end
        end
        
        function train_model(obj, transition_probability_params, train_data, ...
                cells_per_whole_note)
            obj.make_initial_distribution();
            obj.make_transition_model(transition_probability_params);
            obj.make_observation_model(train_data, cells_per_whole_note);
            obj.HMM = HiddenMarkovModel(obj.trans_model, obj.obs_model, ...
                obj.initial_prob);
        end
        
        function make_transition_model(obj, transition_probability_params)
            if strcmp(obj.tm_type, '2015')
                obj.trans_model = BeatTrackingTransitionModel2015(...
                    obj.state_space, transition_probability_params);
            elseif strcmp(obj.tm_type, 'whiteley')
                obj.trans_model = BeatTrackingTransitionModel2006(...
                    obj.state_space, transition_probability_params);
            end
            % Check transition model
            if obj.trans_model.is_corrupt();
                error('Corrupt transition model');
            end
        end
        
        function make_observation_model(obj, train_data, ...
                cells_per_whole_note)
            obj.obs_model = BeatTrackingObservationModelHMM(obj.state_space, ...
                train_data.feature.feat_type, obj.dist_type, ...
                cells_per_whole_note);
            % Train model
            if ~strcmp(obj.dist_type, 'RNN')
                obj.obs_model = obj.obs_model.train_model(train_data);
                obj.train_dataset = train_data.dataset;
            end
        end
        
        function make_initial_distribution(obj)
                n_states = obj.state_space.n_states;
            if obj.use_silence_state
                % always start in the silence state
                obj.initial_prob = zeros(n_states, 1);
                obj.initial_prob(end) = 1;
            else
                if strcmp(obj.tm_type, '2015')
                    obj.initial_prob = ones(n_states, 1) / n_states;
                elseif strcmp(obj.tm_type, 'whiteley')
                    num_state_ids = max(obj.state_space.max_position) * ...
                        obj.state_space.max_n_tempo_states * ...
                        obj.state_space.n_patterns;
                    obj.initial_prob = zeros(num_state_ids, 1);
                    obj.initial_prob(obj.state_space.position_from_state > 0) = ...
                        1 / n_states;
                end
            end
        end
        
        function results = do_inference(obj, y, fname, inference_method, ...
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
%                         try
%                             hidden_state_sequence = ...
%                                 obj.HMM.viterbi(y, obj.use_mex_viterbi);
%                         catch
%                             fprintf(['\n    WARNING: viterbi.cpp has to be', ...
%                                 'compiled, using the pure MATLAB version', ...
%                                 'instead\n']);
                            hidden_state_sequence = obj.HMM.viterbi(y, 0);
%                         end
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
            % strip of silence state
            if obj.use_silence_state
                idx = logical(r_path<=obj.state_space.n_patterns);
            else
                idx = true(length(r_path), 1);
            end
            % compute beat times and bar positions of beats
            meter = zeros(2, length(r_path));
            meter(:, idx) = obj.state_space.meter_from_pattern(r_path(idx), :)';
            beats = obj.find_beat_times(m_path, r_path, y);
            if ~isempty(n_path)
                tempo = obj.state_space.convert_tempo_to_bpm(n_path(idx));
            end
            results{1} = beats;
            results{2} = tempo;
            results{3} = meter;
            results{4} = r_path;
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
                bar_pos_per_beat = obj.state_space.max_position ./ ...
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
                state_pos_per_beat = obj.state_space.max_position ./ ...
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
        
        function beat_positions = get.beat_positions(obj)
            for i_r = 1:obj.state_space.n_patterns
                pos_per_beat = obj.state_space.max_position(i_r) / ...
                    obj.state_space.n_beats_from_pattern(i_r);
                % subtract eps to exclude max_position+1
                obj.beat_positions{i_r} = 1:pos_per_beat:...
                    (obj.state_space.max_position(i_r)+1);
                obj.beat_positions{i_r} = obj.beat_positions{i_r}(1:...
                    obj.state_space.n_beats_from_pattern(i_r));
            end
            beat_positions = obj.beat_positions;
        end
        
    end
    
    
    
    methods (Access=protected)
        
        function beats = find_beat_times(obj, position_state, rhythm_state, ...
                beat_act)
            % [beats] = findBeatTimes(position_state, rhythm_state, param_g)
            %   Find beat times from sequence of bar positions of the HMM beat tracker
            % ----------------------------------------------------------------------
            %INPUT parameter:
            % position_state             : sequence of position states
            % rhythm_state               : sequence of rhythm states
            % beat_act                 : beat activation for correction
            %
            %OUTPUT parameter:
            %
            % beats                    : [nBeats x 2] beat times in [sec] and
            %                           bar.beatnumber
            %
            % 29.7.2012 by Florian Krebs
            % ----------------------------------------------------------------------
            n_frames = length(position_state);
            % set up cell array with beat position for each meter
            beatpositions = obj.beat_positions;
            beats = [];
            if obj.correct_beats
                % resolution of observation model in
                % position_states:
                res_obs = max(obj.state_space.max_position)/obj.max_bar_cells;
                [dist, btype] = max(beatpositions{rhythm_state(1)} - ...
                    position_state(1));
                if (abs(dist) < res_obs/2) && (dist < 0)
                    % if beat is shortly before (within res_obs/2) the audio start we
                    % add one beat at the beginning. E.g. m-sequence starts with
                    % m=2
                    % find audioframe that corresponds to beat
                    % position + res_obs
                    j=1;
                    while position_state(j) < ...
                            (beatpositions{rhythm_state(1)}(btype) + res_obs - 1)
                        j = j + 1;
                    end
                    [~, win_max_offset] = max(beat_act(1:j, ...
                        size(beat_act, 2)));
                    beats = [beats; [win_max_offset, btype]];
                end
            end
            for i = 1:n_frames-1
                if rhythm_state(i) == obj.state_space.n_patterns + 1;
                    % silence state
                    continue;
                end
                for j = 1:length(beatpositions{rhythm_state(i)})
                    beat_pos = beatpositions{rhythm_state(i)}(j);
                    beat_detected = false;
                    if position_state(i) == beat_pos;
                        % current frame = beat frame
                        bt = i;
                        beat_detected = true;
                    elseif ((position_state(i+1) > beat_pos) ...
                            && (position_state(i+1) < position_state(i)))
                        % bar transition between frame i and frame i+1
                        bt = interp1([position_state(i); ...
                            obj.state_space.max_position(rhythm_state(i)) + ...
                            position_state(i+1)], [i; i+1], ...
                            obj.state_space.max_position(rhythm_state(i)) + ...
                            beat_pos);
                        beat_detected = true;
                    elseif ((position_state(i) < beat_pos) ...
                            && (position_state(i+1) > beat_pos))
                        % beat position lies between frame i and frame i+1
                        bt = interp1([position_state(i); position_state(i+1)], ...
                            [i; i+1], beat_pos);
                        beat_detected = true;
                    end
                    if beat_detected
                        if obj.correct_beats
                            % find audioframe that corresponds to beat
                            % position + res_obs
                            max_pos=i;
                            while (max_pos < length(position_state)) ...
                                    && (position_state(max_pos) < ...
                                    (beat_pos + res_obs))
                                max_pos = max_pos + 1;
                            end
                            % find max of last observation feature
                            % TODO: specify which feature to use for
                            % correction
                            [~, win_max_offset] = max(beat_act(floor(bt):max_pos, ...
                                size(beat_act, 2)));
                            bt = win_max_offset + i - 1;
                        end
                        % madmom does not use interpolation. This yields
                        % ~round(bt)
                        beats = [beats; [round(bt), j]];
                        break;
                    end
                end
            end
            if ~isempty(beats)
                % subtract one frame, to have a beat sequence starting at 0
                % seconds.
                beats(:, 1) = (beats(:, 1) ) * obj.frame_length;
            end
        end
        
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
            obj.frame_length = Params.frame_length;
            if isfield(Params, 'observationModelType')
                obj.dist_type = Params.observationModelType;
            else
                obj.dist_type = 'MOG';
            end
            if isfield(Params, 'online')
                obj.max_shift = Params.online.max_shift;
                obj.update_interval = Params.online.update_interval;
                obj.obs_lik_floor = Params.online.obs_lik_floor;
                store_proximity = 1;
            else
                store_proximity = 0;
            end
            obj.use_silence_state = Params.use_silence_state;
            if isfield(Params, 'correct_beats')
                obj.correct_beats = Params.correct_beats;
            else
                obj.correct_beats = 0;
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
            elseif strcmp(obj.tm_type, 'whiteley')
                State_space_params.n_tempi = 30;                
            end
            if strcmp(obj.tm_type, 'whiteley')
                if isfield(Params, 'M')
                    State_space_params.max_positions = Params.M; 
                else
                    State_space_params.max_positions = 1600; 
                end 
            end
            State_space_params.n_patterns = Params.R;
        end
    end
end

