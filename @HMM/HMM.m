classdef HMM
    % Hidden Markov Model Class
    properties
        M                   % number of (max) positions
        Meff                % number of positions per rhythm [R, 1]
        N                   % number of tempo states
        R                   % number of rhythmic pattern states
        rhythm2meter        % assigns each rhythmic pattern to a meter [R x 2]
        rhythm2meter_state  % assigns each rhythmic pattern to a meter state
        rhythm2nbeats       % number of beats per bar [nRhythms x 1]
        % (1, 2, ...) - var not needed anymore but keep due to
        % compatibility
        meter_state2meter   % specifies meter for each meter state (9/8, 8/8, 4/4)
        % [2 x nMeters]  - var not needed anymore but keep due to
        % compatibility
        barGrid             % number of different observation model params
        % of the longest bar (e.g., 64)
        frame_length        % audio frame length in [sec]
        dist_type           % type of parametric distribution
        trans_model         % transition model
        tm_type             % transition model type ('whiteley' or '2015')
        obs_model           % observation model
        initial_prob        % initial state distribution
        init_n_gauss        % number of components of the GMD modelling the initial distribution for each rhythm
        pattern_size        % size of one rhythmical pattern {'beat', 'bar'}
        save_inference_data % save intermediate output of particle filter for visualisation
        inferenceMethod
        viterbi_learning_iterations
        tempo_tying         % 0 = tempo only tied across position states, 1 = global p_n for all changes, 2 = separate p_n for tempo increase and decrease
        n_depends_on_r      % no dependency between n and r
        rhythm_names        % cell array of rhythmic pattern names
        train_dataset       % dataset, which HMM was trained on [string]
        correct_beats       % [0, 1, 2] correct beats after detection
        max_shift           % frame shifts that are allowed in forward path
        obs_lik_floor       % obs_lik has to be > floor to avoid overfitting
        update_interval     % best state sequence is set to the global max each update_interval
        use_silence_state
        pfs                 % transition from silence
        p2s                 % transition to silence
        use_mex_viterbi     % 1: use it, 0: don't use it (~5 times slower)
        beat_positions      % cell array; contains the bar positions of the beats for each rhythm
    end
    
    methods
        function obj = HMM(Params, Clustering)
            if isfield(Params, 'transition_model_type')
                obj.tm_type = Params.transition_model_type;
            else
                obj.tm_type = '2015';
            end
            if strcmp(obj.tm_type, '2015')
                obj.M = max(Clustering.rhythm2nbeats);
            else
                obj.M = Params.M;
            end
            obj.R = Params.R;
            bar_durations = Clustering.rhythm2meter(:, 1) ./ ...
                Clustering.rhythm2meter(:, 2);
            obj.barGrid = max(Params.whole_note_div * bar_durations);
            obj.frame_length = Params.frame_length;
            if isfield(Params, 'observationModelType')
                obj.dist_type = Params.observationModelType;
            else
                obj.dist_type = 'MOG';
            end
            obj.rhythm2meter = Clustering.rhythm2meter;
            obj.rhythm2nbeats = Clustering.rhythm2nbeats;
            % effective number of bar positions per rhythm
            obj.Meff = bar_durations * obj.M ./ (max(bar_durations));
            obj.pattern_size = Params.pattern_size;
            if isfield(Params, 'save_inference_data')
                obj.save_inference_data = Params.save_inference_data;
            else
                obj.save_inference_data = 0;
            end
            if isfield(Params, 'viterbi_learning_iterations') && ...
                    Params.viterbi_learning_iterations > 0
                obj.viterbi_learning_iterations = ...
                    Params.viterbi_learning_iterations;
                obj.tempo_tying = Params.tempo_tying;
            else
                obj.viterbi_learning_iterations = 0;
            end
            if isfield(Params, 'init_n_gauss')
                obj.init_n_gauss = Params.init_n_gauss;
            else
                obj.init_n_gauss = 0;
            end
            if isfield(Params, 'n_depends_on_r')
                obj.n_depends_on_r = Params.n_depends_on_r;
            else
                obj.n_depends_on_r = 1;
            end
            if isfield(Params, 'online')
                obj.max_shift = Params.online.max_shift;
                obj.update_interval = Params.online.update_interval;
                obj.obs_lik_floor = Params.online.obs_lik_floor;
            end
            obj.rhythm_names = Clustering.rhythm_names;
            obj.use_silence_state = Params.use_silence_state;
            if obj.use_silence_state
                obj.p2s = Params.p2s;
                obj.pfs = Params.pfs;
                obj.rhythm_names{obj.R+1} = 'silence';
            end
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
            if isfield(Params, 'N') && strcmp(obj.tm_type, '2015')
                obj.N = Params.N;
            else
                obj.N = nan;
            end
        end
        
        
        function obj = convert_old_model_to_new(obj)
            % This function might be removed
            % in future, but is important for compatibility with older models
            % check dimensions of member variables. This function might be removed
            % in future, but is important for compatibility with older models
            % (in old models Meff and rhythm2meter_state are row vectors
            % [1 x K] but should be column vectors)
            if isempty(obj.Meff) && (size(obj.rhythm2meter, 1) == 1) && ...
                    isempty(obj.rhythm2meter_state)
                obj.rhythm2meter = [obj.rhythm2meter + 2; ...
                    ones(size(obj.rhythm2meter)) * 4];
                obj.obs_model = ...
                    obj.obs_model.convert_to_new_model(obj.rhythm2meter);
                obj.Meff = obj.M * obj.rhythm2meter(1, :) ./ ...
                    obj.rhythm2meter(2, :);
            end
            obj.Meff = obj.Meff(:);
            if length(obj.Meff) ~= obj.R
                obj.Meff = obj.Meff(obj.rhythm2meter_state);
                obj.Meff = obj.Meff(:);
                
            end
            obj.rhythm2meter_state = obj.rhythm2meter_state(:);
            % In old models, pattern change probability was not saved as
            % matrix [RxR]
            if (length(obj.trans_model.pr(:)) == 1) && (obj.R > 1)
                % expand pr to a matrix [R x R]
                % transitions to other patterns
                pr_mat = ones(obj.R, obj.R) * (obj.trans_model.pr / (obj.R-1));
                % pattern self transitions
                pr_mat(logical(eye(obj.R))) = (1 - obj.trans_model.pr);
                obj.trans_model.set_pr(pr_mat);
            end
            if isempty(obj.rhythm2meter)
                obj.rhythm2meter = obj.meter_state2meter(:, ...
                    obj.rhythm2meter_state)';
                obj.obs_model = ...
                    obj.obs_model.convert_to_new_model(obj.rhythm2meter);
            end
            if isempty(obj.rhythm2nbeats)
                % use the denominator of the time signature
                obj.rhythm2nbeats = obj.rhythm2meter(:, 1);
                % replace denominators for compound meters
                % 6/8 has two beats
                obj.rhythm2nbeats(obj.rhythm2nbeats==6) = 2;
                % 9/8 has three beats
                obj.rhythm2nbeats(obj.rhythm2nbeats==9) = 3;
                % 12/8 has four beats
                obj.rhythm2nbeats(obj.rhythm2nbeats==12) = 4;
            end
            obj.tm_type = obj.trans_model.tm_type;
        end
        
        
        function obj = make_transition_model(obj, min_tempo_bpm, max_tempo_bpm, ...
                alpha, pn, pr)
            position_states_per_beat = obj.Meff ./ obj.rhythm2nbeats;
            obj.trans_model = TransitionModel(obj.Meff, ...
                obj.N, obj.R, pn, pr, alpha, position_states_per_beat, ...
                min_tempo_bpm, max_tempo_bpm, obj.frame_length, ...
                obj.use_silence_state, obj.p2s, obj.pfs, ...
                obj.tm_type);
            % set N of the model to N of the transition model
            obj.N = obj.trans_model.N;
            % Check transition model
            if transition_model_is_corrupt(obj.trans_model, 0)
                error('Corrupt transition model');
            end
        end
        
        function obj = make_observation_model(obj, train_data)
            % Create observation model
            obj.obs_model = ObservationModel(obj.dist_type, obj.rhythm2meter, ...
                obj.M, obj.N, obj.R, obj.barGrid, obj.Meff, ...
                train_data.feature.feat_type, obj.use_silence_state, ...
                obj.trans_model.mapping_state_position, ...
                obj.trans_model.mapping_state_rhythm);
            % Train model
            if ~strcmp(obj.dist_type, 'RNN')
                obj.obs_model = obj.obs_model.train_model(train_data);
                obj.train_dataset = train_data.dataset;
            end
        end
        
        function obj = make_initial_distribution(obj, tempo_per_cluster)
            n_states = length(obj.trans_model.mapping_state_position);
            if obj.use_silence_state
                % always start in the silence state
                obj.initial_prob = zeros(n_states, 1);
                obj.initial_prob(end) = 1;
            else
                if obj.init_n_gauss > 0
                    obj.initial_prob = zeros(n_states, 1);
                    for iCluster = 1:size(tempo_per_cluster, 2)
                        meter = obj.rhythm2meter(iCluster, :);
                        tempi = tempo_per_cluster(:, iCluster) * obj.M * obj.frame_length ...
                            / (60 * meter(2));
                        gmm = gmdistribution.fit(tempi(~isnan(tempi)), obj.init_n_gauss);
                        lik = pdf(gmm, (1:obj.N)');
                        % start/stop index of the states that belong to the correct rhythmic pattern
                        startInd = sub2ind([obj.M, obj.N, obj.R], 1, 1, iCluster);
                        stopInd = sub2ind([obj.M, obj.N, obj.R], obj.M, obj.N, iCluster);
                        temp = repmat(lik, 1, obj.M)';
                        obj.initial_prob(startInd:stopInd) = temp(:)./sum(temp(:));
                    end
                    % normalize
                    obj.initial_prob = obj.initial_prob ./ sum(obj.initial_prob);
                else
                    if strcmp(obj.tm_type, 'whiteley')
                        obj.initial_prob = zeros(n_states, 1);
                        % compute number of valid states:
                        n_range = obj.trans_model.maxN - obj.trans_model.minN + ones(obj.R, 1);
                        n_valid_states = obj.Meff(:)' * n_range(:);
                        prob = 1/n_valid_states;
                        for r_i = 1:obj.R
                            for n_i = obj.trans_model.minN(r_i):obj.trans_model.maxN(r_i)
                                start_state = sub2ind([obj.M, obj.trans_model.N, obj.R], 1, ...
                                    n_i, r_i);
                                obj.initial_prob(start_state:start_state+...
                                    obj.Meff(r_i)-1) = prob;
                            end
                        end
                    elseif strcmp(obj.tm_type, '2015')
                        obj.initial_prob = ones(n_states, 1) / n_states;
                    end
                end
            end
        end
        
        function obj = retrain_observation_model(obj, data_file_pattern_barpos_dim, pattern_id)
            %{
            Retrains the observation model for all states corresponding to <pattern_id>.

            :param data_file_pattern_barpos_dim: training data
            :param pattern_id: pattern to be retrained. can be a vector, too.
            :returns: the retrained hmm object

            %}
            obj.obs_model = obj.obs_model.retrain_model(data_file_pattern_barpos_dim, pattern_id);
            %             obj.obs_model.learned_params{pattern_id, :} = ...
            %                 obj.obs_model.fit_distribution(data_file_pattern_barpos_dim(:, pattern_id, :, :));
        end
        
        function results = do_inference(obj, y, fname, inference_method, ...
                belief_func)
            if obj.hmm_is_corrupt
                error('    WARNING: @HMM/do_inference.m: HMM is corrupt\n');
            end
            % compute observation likelihoods
            obs_lik = obj.obs_model.compute_obs_lik(y);
            if strcmp(inference_method, 'HMM_forward')
                % HMM forward path
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
                        try
                            hidden_state_sequence = ...
                                obj.viterbi_decode_mex(obs_lik, fname);
                        catch
                            fprintf('\n    WARNING: viterbi.cpp has to be compiled, using the pure MATLAB version instead\n');
                            hidden_state_sequence = obj.viterbi_decode(obs_lik, fname);
                        end
                    else
                        hidden_state_sequence = obj.viterbi_decode(obs_lik, ...
                            fname);
                    end
                end
            elseif strcmp(inference_method, 'HMM_viterbi_lag')
                hidden_state_sequence = obj.viterbi_fixed_lag_decode(obs_lik, 2);
            else
                error('inference method not specified\n');
            end
            % decode state index into sub indices
            if isprop(obj.trans_model, 'mapping_state_position') && ...
                    ~isempty(obj.trans_model.mapping_state_position)
                m_path = obj.trans_model.mapping_state_position(...
                    hidden_state_sequence)';
                n_path = obj.trans_model.mapping_state_tempo(...
                    hidden_state_sequence)';
                r_path = obj.trans_model.mapping_state_rhythm(...
                    hidden_state_sequence)';
            else
                % For compatibility to old models (created before 2015)
                [m_path, n_path, r_path] = ind2sub([obj.M, obj.N, obj.R], ...
                    hidden_state_sequence(:)');
            end
            % strip of silence state
            if obj.use_silence_state
                idx = logical(r_path<=obj.R);
            else
                idx = true(length(r_path), 1);
            end
            % compute beat times and bar positions of beats
            meter = zeros(2, length(r_path));
            meter(:, idx) = obj.rhythm2meter(r_path(idx), :)';
            beats = obj.find_beat_times(m_path, r_path, y);
            if strcmp(obj.pattern_size, 'bar') && ~isempty(n_path)
                pos_per_beat = obj.Meff(r_path(idx)) ./ ...
                    obj.rhythm2nbeats(r_path(idx));
                tempo = 60 .* n_path(idx)' ./ ...
                    (pos_per_beat * obj.frame_length);
            else
                tempo = 60 .* n_path(idx) / (obj.M * obj.frame_length);
            end
            results{1} = beats;
            results{2} = tempo;
            results{3} = meter;
            results{4} = r_path;
        end
        
        function [obj, bar2cluster] = viterbi_training(obj, features, train_data)
            n_files = length(train_data.file_list);
            A_n = zeros(obj.N * obj.R, obj.N);
            A_r = zeros(obj.R, obj.R);
            observation_per_state = cell(n_files, obj.R, obj.barGrid, features.feat_dim);
            init = zeros(obj.N * obj.R, 1);
            % pattern id that each bar was assigned to in viterbi
            bar2cluster = zeros(size(train_data.bar2cluster));
            for i_file=1:n_files
                [~, fname, ~] = fileparts(train_data.file_list{i_file});
                fprintf('  %i/%i) %s', i_file, n_files, fname);
                % make belief function
                belief_func = obj.make_belief_functions(train_data, i_file);
                % load observations
                observations = features.load_feature(train_data.file_list{i_file});
                obs_lik = obj.obs_model.compute_obs_lik(observations);
                first_frame = max([belief_func{1}(1), 1]);
                best_path = obj.viterbi_iteration(obs_lik, belief_func);
                if isempty(best_path)
                    continue;
                end
                % plot assignment from hidden states to observations after
                % training iteration
                [m_path, n_path, r_path] = ind2sub([obj.M, obj.N, obj.R], best_path(:)');
                % compute beat times and bar positions of beats
                beats = obj.find_beat_times(m_path, r_path, observations);
                beats(:, 1) = beats(:, 1) + (belief_func{1}(1)-1) * obj.frame_length;
                if min(n_path) < 5
                    fprintf('    Low tempo detected at file (n=%i), doing nothing\n', min(n_path));
                end
                % save pattern id per bar
                if isempty(train_data.bar_start_id) % no downbeats annotations available
                    bar2cluster = [];
                else
                    temp = zeros(size(train_data.beats{i_file}, 1), 1);
                    temp(train_data.bar_start_id{i_file}+1) = 1;
                    temp = round(train_data.beats{i_file}(temp & train_data.full_bar_beats{i_file}, 1) / obj.frame_length);
                    if sum(train_data.bar2file == i_file) == length(temp)
                        % make sure temp is between 1 and nFrames
                        temp(temp<1) = 1;
                        temp(temp>length(r_path)) = length(r_path);
                        bar2cluster(train_data.bar2file == i_file) = r_path(temp);
                    else
                        fprintf('    WARNING: incosistency in @HMM/viterbi_training\n');
                    end
                end
                b = ones(length(best_path), 1) * (1:features.feat_dim);
                subs = [repmat(obj.obs_model.state2obs_idx(best_path, 2), features.feat_dim, 1), b(:)];
                % only use observation between first and last observation
                obs = observations(belief_func{1}(1):min([belief_func{1}(end), size(obs_lik, 3) ]), :);
                D = accumarray(subs, obs(:), [], @(x) {x});
                for i_r = unique(r_path(:))'
                    for i_pos = unique(obj.obs_model.state2obs_idx(best_path, 2))'
                        observation_per_state(i_file, i_r, i_pos, :) = D(i_pos, :);
                    end
                end
                for i_frame=2:length(best_path)
                    % count tempo transitions
                    idx1 = ((r_path(i_frame-1) - 1) * obj.N) + n_path(i_frame-1);
                    A_n(idx1, n_path(i_frame)) = A_n(idx1, n_path(i_frame)) + 1;
                    % count pattern transitions (given a bar crossing occured)
                    if m_path(i_frame) < m_path(i_frame-1) % bar crossing
                        A_r(r_path(i_frame-1), r_path(i_frame)) = A_r(r_path(i_frame-1), r_path(i_frame)) + 1;
                    end
                end
                init(((r_path(1) - 1) * obj.N) + n_path(1)) = ...
                    init(((r_path(1) - 1) * obj.N) + n_path(1)) + 1;
            end
            % update initial probabilities
            init = init / sum(init);
            obj.initial_prob = repmat(init(:)', obj.M, 1);
            obj.initial_prob = obj.initial_prob(:);
            % update transition model
            % pattern transitions
            obj.pr = bsxfun(@rdivide, A_r, sum(A_r, 2)); % normalise p_r
            n_times_in_state_ni_at_k_1 = sum(A_n, 2);
            % save tempo transitions
            obj.trans_model.tempo_transition_probs = A_n;
            obj.trans_model.tempo_transition_probs(n_times_in_state_ni_at_k_1>0, :) = bsxfun(@rdivide, A_n(n_times_in_state_ni_at_k_1>0, :), n_times_in_state_ni_at_k_1(n_times_in_state_ni_at_k_1>0));
            if obj.tempo_tying == 0
                A_n(n_times_in_state_ni_at_k_1>0, :) = bsxfun(@rdivide, A_n(n_times_in_state_ni_at_k_1>0, :), n_times_in_state_ni_at_k_1(n_times_in_state_ni_at_k_1>0));
                obj.pn = A_n;
            elseif obj.tempo_tying == 1
                a = 0;
                b = sum(n_times_in_state_ni_at_k_1);
                for i_r=1:obj.R
                    a = a + sum(diag(A_n((i_r-1)*obj.N + 1:i_r*obj.N, :), 0));
                end
                obj.pn = (1 - a/b) / 2;
            elseif obj.tempo_tying == 2
                n_up = 0;
                n_down = 0;
                for i_r=1:obj.R
                    n_up = n_up + sum(diag(A_n((i_r-1)*obj.N + 1:i_r*obj.N, :), 1));
                    n_down = n_down + sum(diag(A_n((i_r-1)*obj.N + 1:i_r*obj.N, :), -1));
                end
                pn_up = n_up / sum(n_times_in_state_ni_at_k_1);
                pn_down = n_down / sum(n_times_in_state_ni_at_k_1);
                obj.pn = [pn_up; pn_down];
            else
                error('specify tempo_tying!\n');
            end
            % find min and max tempo states for each pattern
            for r_i = find(~isnan(obj.pr(:, 1)))'
                obj.minN(r_i) = find(sum(...
                    obj.trans_model.tempo_transition_probs((r_i-1)*obj.N ...
                    + 1:r_i*obj.N, :), 2), 1, 'first');
                obj.maxN(r_i) = find(sum(...
                    obj.trans_model.tempo_transition_probs((r_i-1)*obj.N ...
                    + 1:r_i*obj.N, :), 2), 1, 'last');
            end
            if ~obj.n_depends_on_r % no dependency between n and r
                obj.minN = ones(1, obj.R) * min(obj.minN);
                obj.maxN = ones(1, obj.R) * max(obj.maxN);
                fprintf('    New tempo limits: %i - %i\n', obj.minN(1), obj.maxN(1));
            end
            % update: mew TransitionModel
            obj.trans_model = TransitionModel(obj.M, obj.Meff, obj.N, obj.R, obj.pn, obj.pr, ...
                obj.rhythm2meter, obj.frame_length, obj.minN, obj.maxN);
            % update observation model
            train_data.feats_file_pattern_barPos_dim = observation_per_state;
            obj.obs_model = obj.obs_model.train_model(train_data);
            %  fprintf('  Total log_prob=%.2f\n', sum(log_prob));
        end
        
        function [] = visualise_hidden_states(obj, y, map_sequence, pattern_per_frame)
            % map_sequence: [nFrames x 1] vector
            if nargin < 4
                pattern_per_frame =  obj.obs_model.state2obs_idx(map_sequence, 1);
                bar_pos_per_frame =  obj.obs_model.state2obs_idx(map_sequence, 2);
            else
                bar_pos_per_frame = map_sequence;
            end
            mean_params = obj.obs_model.comp_mean_params();
            pattern = nan(size(y));
            for i=1:size(y, 1)
                if ~isnan(pattern_per_frame(i))
                    pattern(i, 1) =  mean_params(pattern_per_frame(i), ...
                        bar_pos_per_frame(i), 1);
                    pattern(i, 2) =  mean_params(pattern_per_frame(i), ...
                        bar_pos_per_frame(i), 2);
                end
            end
            temp = diff(bar_pos_per_frame);
            temp(abs(temp)>0) = 1;
            temp = [temp(1); temp];
            figure;
            hAxes1=subplot(3, 1, 1);
            stairs(y(:, 2))
            hold on
            stairs(pattern(:, 2), 'r')
            stem(temp*max(y(:, 2)), ':k', 'marker', 'none');
            ylim([min(y(:, 2)), max(y(:, 2))])
            title('Onset feature hi')
            hAxes2=subplot(3, 1, 2);
            stairs(y(:, 1))
            hold on
            stairs(pattern(:, 1), 'r')
            stem(temp*max(y(:, 1)), ':k', 'marker', 'none');
            ylim([min(y(:, 1)), max(y(:, 1))])
            title('Onset feature lo')
            hAxes3=subplot(3, 1, 3);
            stem(temp, 'marker', 'none');
            text(find(temp==1), ones(length(find(temp==1)), 1)*0.5, cellstr(num2str(bar_pos_per_frame(logical(temp==1)))));
            title('Position 64th grid')
            linkaxes([hAxes1,hAxes2,hAxes3], 'x');
        end
        
        
        function save_hmm_data_to_hdf5(obj, folder)
            % saves hmm to hdf5 format to be read into flower.
            % save transition matrix
            transition_model = obj.trans_model.A;
            % save observation model
            % so far only two mixtures implemented
            n_mix = 2;
            feat_dim = length(obj.obs_model.feat_type);
            n_mu = n_mix * feat_dim;
            n_sigma = feat_dim * feat_dim * n_mix;
            n_rows = n_mix + n_mu + n_sigma;
            observation_model = zeros(obj.R * obj.obs_model.barGrid, n_rows);
            for i_r=1:obj.R
                for i_pos=1:obj.obs_model.barGrid_eff(i_r)
                    col = i_pos+(i_r-1)*obj.obs_model.barGrid;
                    temp = obj.obs_model.learned_params{i_r, i_pos}.mu';
                    observation_model(col, 1:n_mu) = temp(:);
                    observation_model(col, (n_mu+1):(n_mu+n_sigma)) = ...
                        obj.obs_model.learned_params{i_r, i_pos}.Sigma(:);
                    observation_model(col, (n_mu+n_sigma+1):n_rows) = ...
                        obj.obs_model.learned_params{i_r, i_pos}.PComponents(:);
                end
            end
            if obj.use_silence_state
                % add the silence state as last state to the model
                col = obj.R * obj.obs_model.barGrid + 1;
                temp = obj.obs_model.learned_params{obj.R+1, 1}.mu';
                observation_model(col, 1:1:n_mu) = temp(:);
                observation_model(col, (n_mu+1):(n_mu+n_sigma)) = ...
                    obj.obs_model.learned_params{obj.R+1, 1}.Sigma(:);
                observation_model(col, (n_mu+n_sigma+1):n_rows) = ...
                    obj.obs_model.learned_params{obj.R+1, 1}.PComponents(:);
            end
            N = obj.trans_model.N;
            M = obj.M;
            R = obj.R;
            P = obj.barGrid;
            state_to_obs = uint8(obj.obs_model.state2obs_idx);
            % rhythm_to_meter: [2 x R]
            rhythm_to_meter = obj.rhythm2meter';
            if size(rhythm_to_meter, 1) ~= 2
                error('Error: rhythm_to_meter wrong dimension\n')
            end
            tempo_ranges = zeros(2, obj.R);
            tempo_ranges(1, :) = obj.trans_model.min_bpm;
            tempo_ranges(2, :) = obj.trans_model.max_bpm;
            num_gmm_mixtures = n_mix;
            obs_feature_dim = feat_dim;
            % transpose to be consistent with C row-major orientation.
            initial_prob = obj.initial_prob;
            mapping_position_state = obj.trans_model.mapping_state_position';
            mapping_tempo_state = obj.trans_model.mapping_state_tempo';
            mapping_rhythm_state = obj.trans_model.mapping_state_rhythm';
            proximity_matrix = obj.trans_model.proximity_matrix';
            observation_model = observation_model;
            state_to_obs = uint8(obj.obs_model.state2obs_idx);
            transition_model_type = obj.tm_type;
            feature_type = obj.obs_model.feat_type{1};
            for i=2:feat_dim
                feature_type = [feature_type, '_', obj.obs_model.feat_type{i}];
            end
            %save to mat file
            save(fullfile(folder, 'robot_hmm_data.mat'), 'M', 'N', 'R', 'P', ...
                'transition_model', 'observation_model', 'initial_prob', ...
                'state_to_obs', 'rhythm_to_meter', 'tempo_ranges', ...
                'num_gmm_mixtures', 'obs_feature_dim', ...
                'mapping_position_state', 'mapping_tempo_state', ...
                'mapping_rhythm_state', 'transition_model_type', ...
                'feature_type','proximity_matrix', '-v7.3');
            fprintf('* Saved model data (Flower) to %s\n', ...
                fullfile(folder, 'robot_hmm_data.mat'));
        end
    end
    
    methods (Access=protected)
        
        function bestpath = viterbi_decode(obj, obs_lik, fname)
            % [ bestpath, delta, loglik ] = viterbi_cont_int( A, obslik, y,
            % initial_prob)
            % Implementation of the Viterbi algorithm
            % ----------------------------------------------------------------------
            %INPUT parameter:
            % obj.trans_model.A     : transition matrix
            % obslik                : observation likelihood [R x nBarGridSize x nFrames]
            % obj.initial_prob      : initial state probabilities
            %
            %OUTPUT parameter:
            % bestpath      : MAP state sequence
            %
            % 26.7.2012 by Florian Krebs
            % ----------------------------------------------------------------------
            
            nFrames = size(obs_lik, 3);
            
            % don't compute states that are irreachable:
            [row, col] = find(obj.trans_model.A);
            maxState = max([row; col]);
            minState = min([row; col]);
            nStates = maxState + 1 - minState;
            
            if obj.save_inference_data,
                x_fac = 10; % decimation factor for x-axis (bar position)
                logP_data = zeros(round(size(obj.trans_model.A, 1) / x_fac), nFrames, 'single');
                best_state = zeros(nFrames, 1);
            end
            delta = obj.initial_prob;
            valid_states = false(maxState, 1);
            valid_states(unique(col)) = true;
            delta(~valid_states) = 0;
            delta = delta(minState:maxState);
            A = obj.trans_model.A(minState:maxState, minState:maxState);
            if length(delta) > 65535
                psi_mat = zeros(nStates, nFrames, 'uint32');  % 32 bit unsigned integer
            else
                psi_mat = zeros(nStates, nFrames, 'uint16'); % 16 bit unsigned integer
            end
            perc = round(0.1*nFrames);
            i_row = 1:nStates;
            j_col = 1:nStates;
            ind = sub2ind([obj.R, obj.barGrid, nFrames ], obj.obs_model.state2obs_idx(minState:maxState, 1), ...
                obj.obs_model.state2obs_idx(minState:maxState, 2), ones(nStates, 1));
            ind_stepsize = obj.barGrid * obj.R;
            O = zeros(nStates, 1);
            validInds = ~isnan(ind); %
            for iFrame = 1:nFrames
                if obj.save_inference_data,
                    for iR=1:obj.R
                        % get start and end state index of current pattern
                        % iR
                        start_ind = sub2ind([obj.M, obj.N, obj.R], 1, 1, iR);
                        end_ind = sub2ind([obj.M, obj.N, obj.R], obj.M, obj.N, iR);
                        % compressed M
                        M_c = round(obj.M / x_fac);
                        % get state indices of current pattern in
                        % compressed states
                        start_ind_c = sub2ind([M_c, obj.N, obj.R], 1, 1, iR);
                        end_ind_c = sub2ind([M_c, obj.N, obj.R], M_c, obj.N, iR);
                        % expand delta
                        if start_ind < minState
                            delta_ex = [zeros(minState-1, 1); delta(1:end_ind-minState+1)];
                        else
                            delta_ex = delta(start_ind-minState+1:end_ind-minState+1);
                        end
                        frame = reshape(full(delta_ex), obj.M, obj.N);
                        % average x_fac blocks
                        frame = conv2(frame, ones(x_fac, 1) ./ x_fac, 'full');
                        % take every x_fac-th block
                        frame = frame(x_fac:x_fac:end, :);
                        logP_data(start_ind_c:end_ind_c, iFrame-1) = log(frame(:));
                        [~, best_state(iFrame-1)] = max(delta);
                        best_state(iFrame-1) = best_state(iFrame-1) + minState - 1;
                    end
                end
                % delta = prob of the best sequence ending in state j at time t, when observing y(1:t)
                % D = matrix of probabilities of best sequences with state i at time
                % t-1 and state j at time t, when bserving y(1:t)
                % create a matrix that has the same value of delta for all entries with
                % the same state i (row)
                % same as repmat(delta, 1, col)
                D = sparse(i_row, j_col, delta(:), nStates, nStates);
                [delta_max, psi_mat(:, iFrame)] = max(D * A);
                % compute likelihood p(yt|x1:t)
                % ind is shifted at each time frame -> all frames are used
                O(validInds) = obs_lik(ind(validInds));
                % increase index to new time frame
                ind = ind + ind_stepsize;
                delta_max = O .* delta_max';
                % normalize
                norm_const = sum(delta_max);
                delta = delta_max / norm_const;
                if rem(iFrame, perc) == 0
                    fprintf('.');
                end
            end
            if obj.save_inference_data,
                % save for visualization
                M = obj.M; N = obj.N; R = obj.R; frame_length = obj.frame_length;
                save(['./', fname, '_hmm.mat'], ...
                    'logP_data', 'M', 'N', 'R', 'frame_length', 'obs_lik', 'x_fac');
            end
            % Backtracing
            bestpath = zeros(nFrames,1);
            [ m, bestpath(nFrames)] = max(delta);
            maxIndex = find(delta == m);
            bestpath(nFrames) = round(median(maxIndex));
            for iFrame=nFrames-1:-1:1
                bestpath(iFrame) = psi_mat(bestpath(iFrame+1),iFrame+1);
            end
            % add state offset
            bestpath = bestpath + minState - 1;
            fprintf(' done\n');
        end
        
        function bestpath = viterbi_fixed_lag_decode(obj, obs_lik, lag)
            % bestpath = viterbi_fixed_lag_decode(obj, obs_lik)
            % Implementation of the Viterbi algorithm with fixed lag
            % ----------------------------------------------------------------------
            %INPUT parameter:
            % obj                   : HMM object
            % obslik                : observation likelihood [R x nBarGridSize x nFrames]
            % lag
            %
            %OUTPUT parameter:
            % bestpath      : MAP state sequence
            %
            % 12.02.2015 by Florian Krebs
            % ----------------------------------------------------------------------
            nFrames = size(obs_lik, 3);
            % don't compute states that are irreachable:
            bestpath = zeros(nFrames,1);
            [row, col] = find(obj.trans_model.A);
            maxState = max([row; col]);
            minState = min([row; col]);
            nStates = maxState + 1 - minState;
            delta = obj.initial_prob;
            valid_states = false(maxState, 1);
            valid_states(unique(col)) = true;
            delta(~valid_states) = 0;
            delta = delta(minState:maxState);
            A = obj.trans_model.A(minState:maxState, minState:maxState);
            if length(delta) > 65535
                psi_mat = zeros(nStates, nFrames, 'uint32');  % 32 bit unsigned integer
            else
                psi_mat = zeros(nStates, nFrames, 'uint16'); % 16 bit unsigned integer
            end
            perc = round(0.1*nFrames);
            i_row = 1:nStates;
            j_col = 1:nStates;
            ind = sub2ind([obj.R, obj.barGrid, nFrames ], ...
                obj.obs_model.state2obs_idx(minState:maxState, 1), ...
                obj.obs_model.state2obs_idx(minState:maxState, 2), ...
                ones(nStates, 1));
            ind_stepsize = obj.barGrid * obj.R;
            O = zeros(nStates, 1);
            validInds = ~isnan(ind);
            for iFrame = 1:nFrames
                % delta = prob of the best sequence ending in state j at time t, when observing y(1:t)
                % D = matrix of probabilities of best sequences with state i at time
                % t-1 and state j at time t, when bserving y(1:t)
                % create a matrix that has the same value of delta for all entries with
                % the same state i (row)
                % same as repmat(delta, 1, col)
                D = sparse(i_row, j_col, delta(:), nStates, nStates);
                [delta_max, psi_mat(:, iFrame)] = max(D * A);
                % compute likelihood p(yt|x1:t)
                % ind is shifted at each time frame -> all frames are used
                O(validInds) = obs_lik(ind(validInds));
                delta_max = O .* delta_max';
                % increase index to new time frame
                ind = ind + ind_stepsize;
                % normalize
                norm_const = sum(delta_max);
                delta = delta_max / norm_const;
                if rem(iFrame, perc) == 0
                    fprintf('.');
                end
                if iFrame > lag
                    [~, best_state_i] = max(delta);
                    % backtracing lag
                    for i = iFrame:-1:(iFrame - lag + 1)
                        best_state_i = psi_mat(best_state_i, i);
                    end
                    % propagate lag frames into future
                    for i = (iFrame - lag + 1):iFrame
                        [~, best_state_i] = max(A(best_state_i, :));
                    end
                    bestpath(iFrame) = best_state_i;
                else
                    [~, bestpath(iFrame)] = max(delta);
                end
            end
            % add state offset
            bestpath = bestpath + minState - 1;
            fprintf(' done\n');
        end
        
        
        function bestpath = viterbi_decode_mex(obj, obs_lik, fname)
            % [ bestpath, delta, loglik ] = viterbi_cont_int( A, obslik, y,
            % initial_prob)
            % Implementation of the Viterbi algorithm
            % ----------------------------------------------------------------------
            %INPUT parameter:
            % obj.trans_model.A     : transition matrix
            % obslik                : observation likelihood [R x nBarGridSize x nFrames]
            % obj.initial_prob      : initial state probabilities
            %
            %OUTPUT parameter:
            % bestpath      : MAP state sequence
            %
            % 26.7.2012 by Florian Krebs
            % ----------------------------------------------------------------------
            [state_ids_i, state_ids_j, trans_prob_ij] = find(obj.trans_model.A);
            % state_ids_j have to be sorted for the mex viterbi.
            % maybe this can be skipped, if find always sorts indices?
            [state_ids_j, idx] = sort(state_ids_j, 'ascend');
            state_ids_i = state_ids_i(idx);
            trans_prob_ij = trans_prob_ij(idx);
            % As there might be wholes in the state spaces, we first compress
            % the state space in order to have lower state ids.
            % mapping from compressed state (cs) to state (s)
            [s_from_cs, ~, state_ids_j] = unique(state_ids_j);
            % mapping from state (s) to compressed state (s)
            cs_from_s = zeros(s_from_cs(end), 1);
            cs_from_s(s_from_cs) = 1:length(s_from_cs);
            % compress states_i
            state_ids_i = cs_from_s(state_ids_i);
            bestpath = obj.viterbi(state_ids_i, state_ids_j, trans_prob_ij, ...
                obj.initial_prob(s_from_cs), obs_lik, ...
                obj.obs_model.state2obs_idx(s_from_cs, :));
            % uncompress states
            bestpath = s_from_cs(bestpath);
            fprintf(' done\n');
        end
        
        function bestpath = viterbi_iteration(obj, obs_lik, belief_func)
            start_frame = max([belief_func{1}(1), 1]); % make sure that belief_func is >= 1
            belief_frames = [belief_func{:, 1}];
            end_frame = min([belief_frames(end), size(obs_lik, 3)]); % make sure that belief_func is < nFrames
            %             nFrames = end_frame - start_frame + 1;
            nFrames = size(obs_lik, 3);
            % don't compute states that are irreachable:
            [row, col] = find(obj.trans_model.A);
            maxState = max([row; col]);
            minState = min([row; col]);
            nStates = maxState + 1 - minState;
            delta = obj.initial_prob(minState:maxState);
            A = obj.trans_model.A(minState:maxState, minState:maxState);
            if length(delta) > 65535
                psi_mat = zeros(nStates, nFrames, 'uint32');  % 32 bit unsigned integer
            else
                psi_mat = zeros(nStates, nFrames, 'uint16'); % 16 bit unsigned integer
            end
            perc = round(0.1*nFrames);
            i_row = 1:nStates;
            j_col = 1:nStates;
            ind = sub2ind([obj.R, obj.barGrid, nFrames ], ...
                obj.obs_model.state2obs_idx(minState:maxState, 1), ...
                obj.obs_model.state2obs_idx(minState:maxState, 2), ...
                ones(nStates, 1));
            ind_stepsize = obj.barGrid * obj.R;
            %             % start index at the first belief function
            %             ind = ind + ind_stepsize * (start_frame-1);
            O = zeros(nStates, 1);
            validInds = ~isnan(ind);
            belief_counter = 1;
            for iFrame = 1:nFrames
                D = sparse(i_row, j_col, delta(:), nStates, nStates);
                [delta_max, psi_mat(:, iFrame)] = max(D * A);
                if iFrame == belief_frames(belief_counter)
                    delta_max = delta_max .* ...
                        belief_func{belief_counter, 2}(minState:maxState)';
                    delta_max = delta_max / sum(delta_max);
                    if sum(isnan(delta_max)) > 0
                        fprintf(' Viterbi path could not be determined (error at frame %i)\n', iFrame);
                        bestpath = [];
                        return
                    end
                    if belief_counter < size(belief_func, 1)
                        belief_counter = belief_counter + 1;
                    end
                end
                % compute likelihood p(yt|x1:t)
                % ind is shifted at each time frame -> all frames are used
                O(validInds) = obs_lik(ind(validInds));
                % increase index to new time frame
                ind = ind + ind_stepsize;
                delta_max = O .* delta_max';
                % normalize
                norm_const = sum(delta_max);
                delta = delta_max / norm_const;
                if rem(iFrame, perc) == 0
                    fprintf('.');
                end
            end
            % Backtracing
            bestpath = zeros(nFrames, 1);
            [m, bestpath(nFrames)] = max(delta);
            % in case there are more than one maximum
            maxIndex = find(delta == m);
            bestpath(nFrames) = round(median(maxIndex));
            for iFrame=nFrames-1:-1:1
                bestpath(iFrame) = psi_mat(bestpath(iFrame+1),iFrame+1);
            end
            % add state offset
            bestpath = bestpath + minState - 1;
            fprintf(' done\n');
        end
        
        
        function [marginal_best_bath, alpha, best_states, minState] = forward_path(obj, ...
                obs_lik, do_output, fname, y)
            % HMM forward path
            store_alpha = 0;
            if obj.max_shift == 0
                do_best_state_selection = 0;
            else
                do_best_state_selection = 1;
            end
            nFrames = size(obs_lik, 3);
            % don't compute states that are irreachable:
            [row, col] = find(obj.trans_model.A);
            maxState = max([row; col]);
            minState = min([row; col]);
            nStates = maxState + 1 - minState;
            A = obj.trans_model.A(minState:maxState, minState:maxState);
            perc = round(0.1*nFrames);
            if obj.use_silence_state
                ind = sub2ind([obj.R+1, obj.barGrid, nFrames ], obj.obs_model.state2obs_idx(minState:maxState, 1), ...
                    obj.obs_model.state2obs_idx(minState:maxState, 2), ones(nStates, 1));
                ind_stepsize = obj.barGrid * (obj.R + 1);
            else
                ind = sub2ind([obj.R, obj.barGrid, nFrames ], obj.obs_model.state2obs_idx(minState:maxState, 1), ...
                    obj.obs_model.state2obs_idx(minState:maxState, 2), ones(nStates, 1));
                ind_stepsize = obj.barGrid * obj.R;
            end
            best_states = zeros(nFrames, 1);
            % incorporate first observation
            O = zeros(nStates, 1);
            validInds = ~isnan(ind);
            O(validInds) = obs_lik(ind(validInds));
            alpha = obj.initial_prob(minState:maxState);
            alpha = A' * alpha;
            alpha = O .* alpha;
            alpha = alpha / sum(alpha);
            [~, best_states(1)] = max(alpha);
            if store_alpha
                % length of recording in frames
                rec_len = nFrames;
                alpha_mat = zeros(nStates, rec_len);
                alpha_mat(:, 1) = log(alpha);
            end
            % move pointer to next observation
            ind = ind + ind_stepsize;
            if do_output, fprintf('    Forward path .'); end
            for iFrame = 2:nFrames
                alpha = A' * alpha;
                % ind is shifted at each time frame -> all frames are used
                O(validInds) = obs_lik(ind(validInds));
                O(validInds & (O < obj.obs_lik_floor)) = obj.obs_lik_floor;
                % increase index to new time frame
                ind = ind + ind_stepsize;
                alpha = O .* alpha;
                % normalize0
                norm_const = sum(alpha);
                alpha = alpha / norm_const;
                if rem(iFrame, perc) == 0 && do_output
                    fprintf('.');
                end
                if (rem(iFrame, obj.update_interval) == 0) || (iFrame < 50)
                    % use global maximum as best state
                    [~, best_states(iFrame)] = max(alpha);
                else
                    C = A(best_states(iFrame-1), :)' .* O;
                    if nnz(C) == 0
                        [~, best_states(iFrame)] = max(alpha);
                    else
                        % find best state among a restricted set of
                        % possible successor states
                        possible_successors = find(A(best_states(iFrame-1), :)) + minState - 1;
                        if do_best_state_selection
                            possible_successors = obj.find_successors2(possible_successors);
                        end
                        possible_successors = possible_successors - minState + 1;
                        [~, idx] = max(alpha(possible_successors));
                        best_states(iFrame) = possible_successors(idx);
                    end
                end
                if store_alpha && (iFrame <= rec_len)
                    alpha_mat(:, iFrame) = log(alpha);
                end
            end
            marginal_best_bath = [];
            % add state offset
            marginal_best_bath = marginal_best_bath + minState - 1;
            
            if do_output, fprintf(' done\n'); end
            if store_alpha
                % save data to file
                m = obj.trans_model.mapping_state_position;
                n = obj.trans_model.mapping_state_tempo;
                r = obj.trans_model.mapping_state_rhythm;
                obs_lik = obs_lik(:, :, 1:rec_len);
                frame_length = obj.frame_length;
                M = obj.M;
                save(['/tmp/', fname, '_hmm_forward.mat'], ...
                    'alpha_mat', 'm', 'n', 'r', 'obs_lik', 'best_states', ...
                    'frame_length', 'M', 'y');
            end
            best_states = best_states + minState - 1;
        end
        
        function beats = find_beat_times(obj, position_state, rhythm_state, beat_act)
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
            numframes = length(position_state);
            % set up cell array with beat position for each meter
            beatpositions = obj.beat_positions;
            beats = [];
            if obj.correct_beats
                % resolution of observation model in
                % position_states:
                res_obs = obj.M/obj.barGrid;
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
            for i = 1:numframes-1
                if rhythm_state(i) == 0
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
                        bt = interp1([position_state(i); obj.Meff(rhythm_state(i)) + position_state(i+1)], ...
                            [i; i+1], obj.Meff(rhythm_state(i)) + beat_pos);
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
                                    (beat_pos + res_obs - 1))
                                max_pos = max_pos + 1;
                            end
                            % find max of last observation feature
                            % TODO: specify which feature to use for
                            % correction
                            [~, win_max_offset] = max(beat_act(floor(bt):max_pos, ...
                                size(beat_act, 2)));
                            bt = win_max_offset + i - 1;
                        end
                        beats = [beats; [round(bt), j]];
                        break;
                    end
                end
            end
            if ~isempty(beats)
                beats(:, 1) = beats(:, 1) * obj.frame_length;
            end
        end
        
        
        
        
        function obs_lik = rnn_format_obs_prob(obj, y)
            obs_lik = zeros(size(y, 2), obj.barGrid, size(y, 1));
            for iR = 1:size(y, 2)
                obs_lik(iR, 1, :) = y(:, iR);
                obs_lik(iR, 2:end, :) = repmat((1-y(:, iR))/(obj.barGrid-1), 1, obj.barGrid-1)';
            end
        end
        
        
        function possible_successors = find_successors(obj, successors)
            m_id = obj.trans_model.mapping_position_state_id(successors)';
            n_id = obj.trans_model.mapping_tempo_state_id(successors)';
            r_id = obj.trans_model.mapping_state_rhythm(successors)';
            % try shifts to both obj.max_shift to the left
            % and right for the position and one tempo up
            % and down
            m_extended = zeros(1, obj.max_shift * 2 + 3);
            n_extended = zeros(1, obj.max_shift * 2 + 3);
            r_extended = zeros(1, obj.max_shift * 2 + 3);
            num_pos = obj.trans_model.num_position_states_per_pattern;
            % -------------------------------------------------
            % old method
            for i_s = find(r_id<=obj.R) % loop over all possible
                %                                 successors excluding the silence state
                % allow position shift for each possible
                % successor
                % define local neighborhood
                n_pos = num_pos{r_id(i_s)}(n_id(i_s));
                % positions before current one
                pos_win = (m_id(i_s) - obj.max_shift):(m_id(i_s) - 1);
                idx = (pos_win < 1);
                pos_win(idx) = (n_pos - sum(idx) + 1):n_pos;
                m_extended(1:obj.max_shift) = pos_win;
                %  positions after current one
                pos_win = (m_id(i_s) + 1):(m_id(i_s) + obj.max_shift);
                idx = (pos_win > n_pos);
                pos_win(idx) = 1:sum(idx);
                m_extended((obj.max_shift + 1):(2 *...
                    obj.max_shift)) = pos_win;
                % original successor state
                m_extended((2 * obj.max_shift + 1)) = m_id(i_s);
                m_curr = obj.trans_model.mapping_state_position(...
                    successors);
                n_extended(1:(2 * obj.max_shift + 1)) = n_id(i_s);
                % tempo up: find closest position state
                if n_id(i_s) < length(num_pos{r_id(i_s)})
                    n = n_id(i_s) + 1;
                    positions_with_tempo_n = ...
                        obj.trans_model.mapping_state_position(...
                        (obj.trans_model.mapping_tempo_state_id == n) & ...
                        (obj.trans_model.mapping_state_rhythm == r_id(i_s)));
                    [~, m_extended(2 * obj.max_shift + 2)] = ...
                        min(abs(positions_with_tempo_n - m_curr));
                    n_extended(2 * obj.max_shift + 2) = n;
                end
                % tempo down: find closest position state
                if n_id(i_s) > 1
                    n = n_id(i_s) - 1;
                    positions_with_tempo_n = ...
                        obj.trans_model.mapping_state_position(...
                        (obj.trans_model.mapping_tempo_state_id == n) & ...
                        (obj.trans_model.mapping_state_rhythm == r_id(i_s)));
                    [~, m_extended(2 * obj.max_shift + 3)] = ...
                        min(abs(positions_with_tempo_n - m_curr));
                    n_extended(2 * obj.max_shift + 3) = n;
                end
                % no pattern shift allowed
                r_extended(1:(2 * obj.max_shift + 3)) = r_id(i_s);
                % remove zero states
                idx = (m_extended > 0);
                m_extended = m_extended(idx);
                n_extended = n_extended(idx);
                r_extended = r_extended(idx);
            end
            possible_successors = zeros(length(m_extended), 1);
            for i_s = 1:length(m_extended)
                possible_successors(i_s) = ...
                    obj.trans_model.mapping_substates_state(...
                    m_extended(i_s), n_extended(i_s), ...
                    r_extended(i_s));
            end
            if sum(r_id > obj.R) > 0
                error('Silence state not yet supported!');
                possible_successors = [possible_successors, sub2ind([obj.M, obj.N, obj.R+1], 1, 1, obj.R+1)];
            end
        end
        
        function possible_successors = find_successors2(obj, successors)
            r_id = obj.trans_model.mapping_state_rhythm(successors)';
            % do not use transitions that go into the silence state
            n_valid_successors = sum(r_id <= obj.R);
            extended_states = zeros(1, n_valid_successors * 6);
            p = 1;
            for i_s = find(r_id <= obj.R) % loop over all possible
                extended_states(p:p+5)= obj.trans_model.proximity_matrix(...
                    successors(i_s), :);
                p = p + 6;
            end
            extended_states = unique(extended_states(...
                extended_states > 0));
            possible_successors = [successors, ...
                extended_states];
        end
        
        function hmm_corrupt = hmm_is_corrupt(obj)
            num_states_hypothesis = [length(obj.initial_prob);
                size(obj.obs_model.state2obs_idx, 1);
                size(obj.trans_model.A, 1);
                length(obj.trans_model.mapping_state_tempo);
                length(obj.trans_model.mapping_state_position);
                length(obj.trans_model.mapping_state_rhythm);
                length(obj.trans_model.mapping_tempo_state_id)];
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
    end
    
    methods
        function belief_func = make_belief_function(obj, Constraint)
            for c=1:length(Constraint.type)
                if strcmp(Constraint.type{c}, 'downbeats')
                    tol_downbeats = 0.05; % given in beat proportions
                    % compute tol_win in [frames]
                    % belief_func:
                    % col1: frames where annotation is available,
                    % col2: sparse vector that is one for possible states
                    belief_func = cell(length(Constraint.data{c}), 2);
                    beatpositions = obj.beat_positions;
                    bar_pos_per_beat = obj.Meff ./ obj.rhythm2nbeats;
                    win_pos = tol_downbeats .* bar_pos_per_beat;
                    for i_db = 1:length(Constraint.data{c})
                        i_frame = max([1, round(Constraint.data{c}(i_db) / ...
                            obj.frame_length)]);
                        belief_func{i_db, 1} = i_frame;
                        idx = false(obj.trans_model.num_states, 1); 
                        % how many bar positions are one beat?
                        for i_r = 1:obj.R
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
                elseif strcmp(Constraint.type{c}, 'beats')
                    tol_beats = 0.03; % given in beat proportions
                    beatpositions = obj.beat_positions;
                    % find states which are considered in the window
                    idx = false(obj.trans_model.num_states, 1); 
                    for i_r = 1:length(obj.R)
                        bar_pos_per_beat = obj.Meff(i_r) ./ ...
                            obj.rhythm2nbeats(i_r);
                        win_pos = tol_beats * bar_pos_per_beat;
                        for i_b = 1:length(beatpositions{i_r})
                            win_left = obj.trans_model.mapping_state_position > ...
                                    (beatpositions{i_r}(i_b) - win_pos);
                            win_right = obj.trans_model.mapping_state_position < ...
                                    (beatpositions{i_r}(i_b) + win_pos);    
                            idx = idx | (win_left & win_right);
                        end
                    end
                    belief_func = cell(length(Constraint.data{c}), 2);
                    for i_db = 1:length(Constraint.data{c})
                        i_frame = max([1, round(Constraint.data{c}(i_db) / ...
                            obj.frame_length)]);
                        belief_func{i_db, 1} = i_frame;
                        belief_func{i_db, 2} = idx;
                    end
                elseif strcmp(Constraint.type{c}, 'meter')
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
        
        function beat_positions = get.beat_positions(obj)
            for i_r = 1:obj.R
                pos_per_beat = obj.Meff(i_r) / obj.rhythm2nbeats(i_r);
                obj.beat_positions{i_r} = 1:pos_per_beat:obj.Meff(i_r);
            end
            beat_positions = obj.beat_positions;
        end
        
    end
    
    methods (Static)
        
        [m, n] = getpath(M, annots, frame_length, nFrames);
        
        [bestpath] = viterbi(state_ids_i, state_ids_j, trans_prob_ij, ...
            initial_prob, obs_lik, state2obs_idx);
        
    end
    
    
end
