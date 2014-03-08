classdef HMM
    % Hidden Markov Model Class
    properties (SetAccess=private)
        M                   % number of (max) positions
        Meff                % number of positions per meter
        N                   % number of tempo states
        R                   % number of rhythmic pattern states
        T                   % number of different meter
        pn                  % probability of a switch in tempo
        pr                  % probability of a switch in rhythmic pattern
        rhythm2meter        % assigns each rhythmic pattern to a meter state (1, 2, ...)
        meter_state2meter   % specifies meter for each meter state (9/8, 8/8, 4/4)
        barGrid             % number of different observation model params per bar (e.g., 64)
        minN                % min tempo (n_min) for each rhythmic pattern
        maxN                % max tempo (n_max) for each rhythmic pattern
        frame_length        % audio frame length in [sec]
        dist_type           % type of parametric distribution
        trans_model         % transition model
        obs_model           % observation model
        initial_prob        % initial state distribution
        init_n_gauss        % number of components of the GMD modelling the initial distribution for each rhythm
        pattern_size        % size of one rhythmical pattern {'beat', 'bar'}
        save_inference_data % save intermediate output of particle filter for visualisation
        inferenceMethod
        tempo_tying         % 0 = tempo only tied across position states, 1 = global p_n for all changes, 2 = separate p_n for tempo increase and decrease
        viterbi_learning_iterations 
        n_depends_on_r      % no dependency between n and r
        
    end
    
    methods
        function obj = HMM(Params, rhythm2meter)
            
            obj.M = Params.M;
            obj.Meff = Params.Meff;
            obj.N = Params.N;
            obj.R = Params.R;
            obj.pn = Params.pn;
            if isfield(Params, 'cluster_transitions_fln') && exist(Params.cluster_transitions_fln, 'file')
                obj.pr = dlmread(Params.cluster_transitions_fln);
            else
                obj.pr = Params.pr;
            end
            obj.barGrid = max(Params.barGrid_eff);
            obj.frame_length = Params.frame_length;
            obj.dist_type = Params.observationModelType;
            obj.init_n_gauss = Params.init_n_gauss;
            obj.rhythm2meter = rhythm2meter;
            obj.meter_state2meter = Params.meters;
            obj.pattern_size = Params.pattern_size;
            obj.save_inference_data = Params.save_inference_data;
            obj.inferenceMethod = Params.inferenceMethod;
            obj.tempo_tying = Params.tempo_tying;
            obj.viterbi_learning_iterations = Params.viterbi_learning_iterations;
            obj.n_depends_on_r = Params.n_depends_on_r;
        end
        
        function obj = make_transition_model(obj, minTempo, maxTempo)
            
            % convert from BPM into barpositions / audio frame
            meter_num = obj.meter_state2meter(1, obj.rhythm2meter);

            if strcmp(obj.pattern_size, 'bar')
                obj.minN = floor(obj.Meff(obj.rhythm2meter) .* obj.frame_length .* minTempo ./ (meter_num * 60));
                obj.maxN = ceil(obj.Meff(obj.rhythm2meter) .* obj.frame_length .* maxTempo ./ (meter_num * 60));
            else
                obj.minN = floor(obj.M * obj.frame_length * minTempo ./ 60);
                obj.maxN = ceil(obj.M * obj.frame_length * maxTempo ./ 60);
            end            

            if max(obj.maxN) > obj.N
                fprintf('N should be %i instead of %i\n', max(obj.maxN), obj.N); 
            end
            
            if ~obj.n_depends_on_r % no dependency between n and r
                obj.minN = ones(1, obj.R) * min(obj.minN);
                obj.maxN = ones(1, obj.R) * max(obj.maxN);
            end
            
            % Create transition model
            if obj.viterbi_learning_iterations > 0 % for viterbi learning use uniform tempo prior
                obj.minN = ones(1, obj.R);
                obj.maxN = ones(1, obj.R) * obj.N;
            end

            obj.trans_model = TransitionModel(obj.M, obj.Meff, obj.N, obj.R, obj.pn, obj.pr, ...
                obj.rhythm2meter, obj.minN, obj.maxN);

            % Check transition model
            if transition_model_is_corrupt(obj.trans_model, 0)
                error('Corrupt transition model');
            end
            
        end
        
        function obj = make_observation_model(obj, data_file_pattern_barpos_dim)
            
            % Create observation model
            obj.obs_model = ObservationModel(obj.dist_type, obj.rhythm2meter, ...
                obj.meter_state2meter, obj.M, obj.N, obj.R, obj.barGrid, obj.Meff);
            
            % Train model
            obj.obs_model = obj.obs_model.train_model(data_file_pattern_barpos_dim);
            
        end
        
        function obj = make_initial_distribution(obj, tempo_per_cluster)
            n_states = obj.M * obj.N * obj.R;
            if obj.init_n_gauss > 0
                obj.initial_prob = zeros(n_states, 1);
                for iCluster = 1:size(tempo_per_cluster, 2)
                    meter = obj.meter_state2meter(:, obj.rhythm2meter(iCluster));
                    tempi = tempo_per_cluster(:, iCluster) * obj.M * obj.frame_length ...
                        / (60 * meter(2));
                    gmm = gmdistribution.fit(tempi(~isnan(tempi)), obj.init_n_gauss);
                    %                     gmm_wide = gmdistribution(gmm.mu, gmm.Sigma, gmm.PComponents);
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
                obj.initial_prob = ones(n_states, 1) / n_states;
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
        
        function [beats, tempo, rhythm, meter] = do_inference(obj, y, fname)
            
            % compute observation likelihoods
            obs_lik = obj.obs_model.compute_obs_lik(y);
            
            
            if strcmp(obj.inferenceMethod, 'HMM_forward')
                % HMM forward path
                [hidden_state_sequence, ~, psi, min_state] = obj.forward_path(obs_lik, fname);
                
            elseif strcmp(obj.inferenceMethod, 'HMM_viterbi')
                % decode MAP state sequence using Viterbi
                hidden_state_sequence = obj.viterbi_decode(obs_lik, fname);
            else
                error('inference method not specified\n');
            end
            
            % factorial HMM: mega states -> substates
            [m_path, n_path, r_path] = ind2sub([obj.M, obj.N, obj.R], hidden_state_sequence(:)');
            
            if strcmp(obj.inferenceMethod, 'HMM_forward')
                [m_path, n_path, r_path] = obj.refine_forward_path(m_path, n_path, r_path, psi, min_state);
            end
            
            %             dets=[m_path(:), n_path(:), r_path(:)];
            %             mean_params = obj.obs_model.comp_mean_params;
            %             ind = sub2ind([obj.R, obj.barGrid], r_path, obj.obs_model.state2obs_idx(hidden_state_sequence, 2));
            %             mean_params = mean_params(ind);
            %             save(['./temp/', fname, '_map.mat'], 'dets', 'y', 'mean_params');
            % meter path
            t_path = obj.rhythm2meter(r_path);
            % compute beat times and bar positions of beats
            meter = obj.meter_state2meter(:, t_path);
            beats = obj.find_beat_times(m_path, t_path, n_path);
            tempo = meter(2, :) .* 60 .* n_path / (obj.M * obj.frame_length);
            rhythm = r_path;
            
            
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
                fprintf('%i/%i) %s', i_file, n_files, fname);
                % make belief function
                belief_func = train_data.make_belief_functions(obj, i_file);
                % load observations
                observations = features.load_feature(train_data.file_list{i_file});
                obs_lik = obj.obs_model.compute_obs_lik(observations);
                best_path = obj.viterbi_iteration(obs_lik, belief_func);
                if isempty(best_path)
                    continue;
                end
                [m_path, n_path, r_path] = ind2sub([obj.M, obj.N, obj.R], best_path(:)');
                if min(n_path) < 5
                   fprintf('    Low tempo detected at file (n=%i), ignoring file.\n', min(n_path)); 
                   continue;
                end
% %                 % compute beat times and bar positions of beats
%                 t_path = obj.rhythm2meter(r_path);
%                 beats = obj.find_beat_times(m_path, t_path, n_path);
%                 beats(:, 1) = beats(:, 1) + (belief_func{1}(1)-1) * obj.frame_length;
%                 BeatTracker.save_beats(beats, ['temp/', fname, '.txt']);
                
                % save pattern id per bar
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
                    obj.minN(r_i) = find(sum(obj.trans_model.tempo_transition_probs((r_i-1)*obj.N + 1:r_i*obj.N, :), 2), 1, 'first');
                    obj.maxN(r_i) = find(sum(obj.trans_model.tempo_transition_probs((r_i-1)*obj.N + 1:r_i*obj.N, :), 2), 1, 'last');
            end
            if ~obj.n_depends_on_r % no dependency between n and r
                obj.minN = ones(1, obj.R) * min(obj.minN);
                obj.maxN = ones(1, obj.R) * max(obj.maxN);
            end
            obj.trans_model = TransitionModel(obj.M, obj.Meff, obj.N, obj.R, obj.pn, obj.pr, ...
                obj.rhythm2meter, obj.minN, obj.maxN);          
            % update observation model
            obj.obs_model = obj.obs_model.train_model(observation_per_state);
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
            
            delta = obj.initial_prob(minState:maxState);
            A = obj.trans_model.A(minState:maxState, minState:maxState);
            if length(delta) > 65535
                %     fprintf('    Size of Psi = %.1f MB\n', maxState * nFrames * 4 / 10^6);
                psi_mat = zeros(nStates, nFrames, 'uint32');  % 32 bit unsigned integer
            else
                %     fprintf('    Size of Psi = %.1f MB\n', maxState * nFrames * 2 / 10^6);
                psi_mat = zeros(nStates, nFrames, 'uint16'); % 16 bit unsigned integer
            end
            perc = round(0.1*nFrames);
            i_row = 1:nStates;
            j_col = 1:nStates;
            ind = sub2ind([obj.R, obj.barGrid, nFrames ], obj.obs_model.state2obs_idx(minState:maxState, 1), ...
                obj.obs_model.state2obs_idx(minState:maxState, 2), ones(nStates, 1));
            ind_stepsize = obj.barGrid * obj.R;
            
            % incorporate first observation
            O = zeros(nStates, 1);
            validInds = ~isnan(ind);
            O(validInds) = obs_lik(ind(validInds));
            delta = O .* delta;
            delta = delta / sum(delta);
            % move pointer to next observation
            ind = ind + ind_stepsize;
            fprintf('    Decoding (viterbi) .');
            
            for iFrame = 2:nFrames
                if obj.save_inference_data,
                    for iR=1:obj.R
                        start_ind = sub2ind([obj.M, obj.N, obj.R], 1, 1, iR);
                        end_ind = sub2ind([obj.M, obj.N, obj.R], obj.M, obj.N, iR);
                        M_c = round(obj.M / x_fac);
                        start_ind_c = sub2ind([M_c, obj.N, obj.R], 1, 1, iR);
                        end_ind_c = sub2ind([M_c, obj.N, obj.R], M_c, obj.N, iR);
                        % expand delta
                        if start_ind < minState
                            delta_ex = [zeros(minState-1, 1); delta(1:end_ind-minState+1)];
                        else
                            delta_ex = delta(start_ind-minState+1:end_ind-minState+1);
                        end
                        %                         frame = imresize(reshape(full(delta_ex), obj.M, obj.N), [M_c, obj.N]);
                        frame = reshape(full(delta_ex), obj.M, obj.N);
                        % average x_fac blocks
                        frame = conv2(frame, ones(x_fac, 1) ./ x_fac, 'full');
                        % take every x_fac-th block
                        frame = frame(x_fac:x_fac:end, :);
                        logP_data(start_ind_c:end_ind_c, iFrame-1) = log(frame(:));
                        [~, best_state(iFrame-1)] = max(delta);
                        %                         best_state(iFrame-1) = best_state(iFrame-1) + minState - 1;
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
                O = zeros(nStates, 1);
                validInds = ~isnan(ind);
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
                save(['~/diss/src/matlab/beat_tracking/bayes_beat/temp/', fname, '_hmm.mat'], ...
                    'logP_data', 'M', 'N', 'R', 'frame_length', 'obs_lik', 'x_fac');
                
                %                 save(['~/diss/src/matlab/beat_tracking/bayes_beat/temp/', fname, '_hmm.mat'], ...
                %                     'logP_data', 'M', 'N', 'R', 'frame_length', 'obs_lik', 'x_fac', 'psi_mat', 'best_state', 'minState');
            end
            
            % Backtracing
            bestpath = zeros(nFrames,1);
            [ ~, bestpath(nFrames)] = max(delta);
            for iFrame=nFrames-1:-1:1
                bestpath(iFrame) = psi_mat(bestpath(iFrame+1),iFrame+1);
            end
            
            % add state offset
            bestpath = bestpath + minState - 1;
            fprintf(' done\n');
        end
        
        function bestpath = viterbi_decode_log(obj, obs_lik)
            % [ bestpath, delta, loglik ] = viterbi_cont_int( A, obslik, y,
            % initial_prob)
            % Implementation of the Viterbi algorithm
            % ----------------------------------------------------------------------
            %INPUT parameter:
            % A             : transition matrix
            % obslik        : structure containing the observation model
            % initial_prob   : initial state probabilities
            %
            %OUTPUT parameter:
            % bestpath      : MAP state sequence
            % delta         : p(x_T | y_1:T)
            % loglik        : p(y_t | y_1:t-1)
            %               (likelihood of the sequence p(y_1:T) would be prod(loglik)
            %
            % 26.7.2012 by Florian Krebs
            % ----------------------------------------------------------------------
            nFrames = size(obs_lik, 3);
            loglik = zeros(nFrames, 1);
            [row, col] = find(obj.trans_model.A);
            
            maxState = max([row; col]);
            minState = min([row; col]);
            nStates = maxState + 1 - minState;
            i_row_lin = 1:nStates;
            j_col_lin = 1:nStates;
            delta_lin = obj.initial_prob(minState:maxState);
            delta = log(obj.initial_prob(minState:maxState));
            A_lin = obj.trans_model.A(minState:maxState, minState:maxState);
            if length(delta) > 65535
                %     fprintf('    Size of Psi = %.1f MB\n', maxState * nFrames * 4 / 10^6);
                psi_mat = zeros(nStates, nFrames, 'uint32');  % 32 bit unsigned integer
            else
                %     fprintf('    Size of Psi = %.1f MB\n', maxState * nFrames * 2 / 10^6);
                psi_mat = zeros(nStates, nFrames, 'uint16'); % 16 bit unsigned integer
            end
            
            %             alpha = zeros(nFrames, 1); % most probable state for each frame given by forward path
            perc = round(0.1*nFrames);
            
            ind = sub2ind([obj.R, obj.barGrid, nFrames ], obj.obs_model.state2obs_idx(minState:maxState, 1), ...
                obj.obs_model.state2obs_idx(minState:maxState, 2), ones(nStates, 1));
            ind_stepsize = obj.barGrid * obj.R;
            
            fprintf('    Decoding (viterbi) .');
            %             logP_data = sparse(size(A, 1), nFrames);
            delta_max = -inf(size(delta));
            A_log = A_lin;
            A_log(find(A_log)) = log(A_log(find(A_log)));
            [i_row, j_col] = find(A_log);
            for iFrame = 1:nFrames
                % linear
                D = sparse(i_row_lin, j_col_lin, delta_lin(:), nStates, nStates);
                [delta_max_lin, psi_mat(:,iFrame)] = max(D * A_lin);
                % log
                D = sparse(i_row, j_col, delta(i_row), nStates, nStates);
                X = A_log + D;
                delta_max(1:max(j_col)) = accumarray(j_col, X(sub2ind([size(X)], i_row, j_col)), [], @max, 0);
                delta_max(delta_max==0) = -inf;
                %                 argmax = @(x) find(x==max(x));
                %
                %                 psi = zeros(size(delta));
                %                 for i=1:length(a)
                %                     ind = (j_col == a(i));
                %                     [temp, psi(i)] = max(X(sub2ind([size(X)], i_row(ind), j_col(ind))));
                %                     psi(i) = psi(i) + find(ind, 1) - 1;
                %                 end
                %                 psi_mat(:,iFrame) = i_row(psi);
                %                 delta_max2 = accumarray(j_col, X(sub2ind([size(X)], i_row, j_col)), [], @(x) find(x==max(x)), 0);
                %                 [delta_max, psi_mat(:,iFrame)] = max(A + D);
                % compute likelihood p(yt|x1:t)
                O = zeros(nStates, 1);
                validInds = ~isnan(ind);
                O(validInds) = obs_lik(ind(validInds));
                % increase index to new time frame
                ind = ind + ind_stepsize;
                delta_lin = O .* delta_max_lin';
                delta = log(O) + delta_max;
                
                % normalize
                norm_const = sum(delta_lin);
                delta_lin = delta_lin / norm_const;
                if rem(iFrame, perc) == 0
                    fprintf('.');
                end
            end
            
            % Backtracing
            bestpath = zeros(nFrames,1);
            [ ~, bestpath(nFrames)] = max(delta);
            for iFrame=nFrames-1:-1:1
                bestpath(iFrame) = psi_mat(bestpath(iFrame+1),iFrame+1);
            end
            
            % add state offset
            bestpath = bestpath + minState - 1;
            
            fprintf(' done\n');
            
        end
        
        function bestpath = viterbi_iteration(obj, obs_lik, belief_func)
            start_frame = max([belief_func{1}(1), 1]); % make sure that belief_func is >= 1
            end_frame = min([belief_func{1}(end), size(obs_lik, 3)]); % make sure that belief_func is < nFrames
            nFrames = end_frame - start_frame + 1;
            % don't compute states that are irreachable:
            [row, col] = find(obj.trans_model.A);
            maxState = max([row; col]);
            minState = min([row; col]);
            nStates = maxState + 1 - minState;
            
            delta = obj.initial_prob(minState:maxState);
            A = obj.trans_model.A(minState:maxState, minState:maxState);
            if length(delta) > 65535
                %     fprintf('    Size of Psi = %.1f MB\n', maxState * nFrames * 4 / 10^6);
                psi_mat = zeros(nStates, nFrames, 'uint32');  % 32 bit unsigned integer
            else
                %     fprintf('    Size of Psi = %.1f MB\n', maxState * nFrames * 2 / 10^6);
                psi_mat = zeros(nStates, nFrames, 'uint16'); % 16 bit unsigned integer
            end
            perc = round(0.1*nFrames);
            i_row = 1:nStates;
            j_col = 1:nStates;
            ind = sub2ind(size(obs_lik), obj.obs_model.state2obs_idx(minState:maxState, 1), ...
                obj.obs_model.state2obs_idx(minState:maxState, 2), ones(nStates, 1));
            ind_stepsize = obj.barGrid * obj.R;
            % start index at the first belief function
            ind = ind + ind_stepsize * (start_frame-1);
             
            fprintf('    Decoding (viterbi training) .');

            for iFrame = 1:nFrames
                D = sparse(i_row, j_col, delta(:), nStates, nStates);
                [delta_max, psi_mat(:, iFrame)] = max(D * A);
                if sum(belief_func{1} == iFrame+start_frame-1)
                    delta_max = delta_max .* belief_func{2}(belief_func{1} == iFrame+start_frame-1, minState:maxState);
                    delta_max = delta_max / sum(delta_max);
                    if sum(isnan(delta_max)) > 0
                        fprintf(' Viterbi path could not be determined (error at beat %i)\n', find(belief_func{1} == iFrame));
                        bestpath = [];
                        return
                    end
                end
                
                % compute likelihood p(yt|x1:t)
                O = zeros(nStates, 1);
                validInds = ~isnan(ind);
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
            bestpath = zeros(nFrames,1);
            [ ~, bestpath(nFrames)] = max(delta);
            for iFrame=nFrames-1:-1:1
                bestpath(iFrame) = psi_mat(bestpath(iFrame+1),iFrame+1);
            end
            
            % add state offset
            bestpath = bestpath + minState - 1;
            fprintf(' done\n');
        end
        
        function [bestpath, alpha, psi, minState] = forward_path(obj, obs_lik, fname)
            % HMM forward path
            
            nFrames = size(obs_lik, 3);
            
            % don't compute states that are irreachable:
            [row, col] = find(obj.trans_model.A);
            maxState = max([row; col]);
            minState = min([row; col]);
            nStates = maxState + 1 - minState;
            
            A = obj.trans_model.A(minState:maxState, minState:maxState);
            i_row = 1:nStates;
            j_col = 1:nStates;
            
            psi = zeros(nStates, nFrames, 'uint16'); % 16 bit unsigned integer
            
            alpha = sparse(nStates, nFrames);
            alpha(:, 1) = obj.initial_prob(minState:maxState);
            
            perc = round(0.1*nFrames);
            ind = sub2ind([obj.R, obj.barGrid, nFrames ], obj.obs_model.state2obs_idx(minState:maxState, 1), ...
                obj.obs_model.state2obs_idx(minState:maxState, 2), ones(nStates, 1));
            ind_stepsize = obj.barGrid * obj.R;
            
            % incorporate first observation
            O = zeros(nStates, 1);
            validInds = ~isnan(ind);
            O(validInds) = obs_lik(ind(validInds));
            alpha(:, 1) = O .* alpha(:, 1);
            alpha(:, 1) = alpha(:, 1) / sum(alpha(:, 1));
            % move pointer to next observation
            ind = ind + ind_stepsize;
            fprintf('    Forward path .');
            
            for iFrame = 2:nFrames
                % delta = prob of the best sequence ending in state j at time t, when observing y(1:t)
                % D = matrix of probabilities of best sequences with state i at time
                % t-1 and state j at time t, when bserving y(1:t)
                % create a matrix that has the same value of delta for all entries with
                % the same state i (row)
                % same as repmat(delta, 1, col)
                alpha(:, iFrame) = A' * alpha(:, iFrame-1);
                D = sparse(i_row, j_col, alpha(:, iFrame), nStates, nStates);
                [~, psi(:, iFrame)] = max(D * A);
                %                 [ ~, psi(:, iFrame)] = max(bsxfun(@times, A, alpha(:, iFrame-1)));
                % compute likelihood p(yt|x1:t)
                O = zeros(nStates, 1);
                validInds = ~isnan(ind);
                % ind is shifted at each time frame -> all frames are used
                O(validInds) = obs_lik(ind(validInds));
                % increase index to new time frame
                ind = ind + ind_stepsize;
                alpha(:, iFrame) = O .* alpha(:, iFrame);
                % normalize
                norm_const = sum(alpha(:, iFrame));
                alpha(:, iFrame) = alpha(:, iFrame) / norm_const;
                if rem(iFrame, perc) == 0
                    fprintf('.');
                end
            end
            
            [~, bestpath] = max(alpha);
            
            % add state offset
            bestpath = bestpath + minState - 1;
            psi = psi + minState - 1;
            fprintf(' done\n');
        end
        
        function [m_path_new, n_path_new, r_path_new] = refine_forward_path(obj, m_path, n_path, r_path, psi, min_state)
            addpath('~/diss/src/matlab/libs/matlab_utils')
            m_path_new = m_path;
            n_path_new = n_path;
            r_path_new = r_path;
            % Wait for 3 seconds
            c=0;
            par.A = [1, 1; 0, 1];
            par.Q = [0.1, 0; 0, 0.001];
            par.C = [1, 0];
            par.R = 300;
            P = ones(2, 2);
            x = [m_path_new(150-1); n_path_new(150-1)];
            for iFrame = 150:length(m_path)
                if abs(m_path(iFrame) - x(1)) < abs(m_path(iFrame) + obj.Meff(r_path(iFrame))- x(1))
                    y = m_path(iFrame);
                else
                    y = m_path(iFrame) + obj.Meff(r_path(iFrame));
                end
                %                 y = min([abs(m_path(iFrame) - x(1)), ...
                %                         abs(m_path(iFrame) + obj.Meff(r_path(iFrame))- x(1))]) + ...
                %                         x(1);
                
                [ x, P, likelihood, P2, E, x_pred ] = KF( x, P, y, par);
                m_path_new(iFrame) = mod(x(1) - 1, obj.Meff(r_path(iFrame))) + 1;
                x(1) = mod(x(1) - 1, obj.Meff(r_path(iFrame))) + 1;
                n_path_new(iFrame) = x(2);
                %                 if m_path(iFrame) == mod(m_path(iFrame-1) + n_path(iFrame-1) - 1, obj.Meff(r_path(iFrame))) + 1% valid path
                %                     c = c + 1;
                %                     if c < 30
                %                         % stay at old path
                %                         % find linear index of old state
                %                         [j_old] = sub2ind([obj.M, obj.N, obj.R], ...
                %                             m_path_new(iFrame-1), n_path_new(iFrame-1), r_path(iFrame-1));
                %                         % find most probable successor of old state
                %                         j_max = find(psi(:, iFrame) == j_old, 1) + min_state - 1;
                %                         if isempty(j_max)
                %                             m_path_new(iFrame) =  mod(m_path_new(iFrame-1) + n_path_new(iFrame-1) - 1, obj.Meff(r_path(iFrame))) + 1;
                %                             n_path_new(iFrame) = n_path_new(iFrame-1);
                %                         else
                %                             [m_path_new(iFrame), n_path_new(iFrame), r_path_new(iFrame)] = ind2sub([obj.M, obj.N, obj.R], j_max);
                %                         end
                %
                %                     else
                %                         m_path_new(iFrame) = m_path(iFrame);
                %                         n_path(iFrame) = n_path(iFrame);
                %                     end
                %                 else
                %                     c = 0;
                %                     % stay at old path
                %                     % find linear index of old state
                %                     [j_old] = sub2ind([obj.M, obj.N, obj.R], ...
                %                         m_path_new(iFrame-1), n_path_new(iFrame-1), r_path(iFrame-1));
                %                     % find most probable successor of old state
                %                     j_max = find(psi(:, iFrame) == j_old, 1) + min_state - 1;
                %                     if isempty(j_max)
                %                         m_path_new(iFrame) =  mod(m_path_new(iFrame-1) + n_path_new(iFrame-1) - 1, obj.Meff(r_path(iFrame))) + 1;
                %                         n_path_new(iFrame) = n_path_new(iFrame-1);
                %                     else
                %                         [m_path_new(iFrame), n_path_new(iFrame), r_path_new(iFrame)] = ind2sub([obj.M, obj.N, obj.R], j_max);
                %                     end
                %                 end
                
            end
            figure; plot(m_path); hold on; plot(m_path_new(1:iFrame), 'r');
        end
        
        function beats = find_beat_times(obj, positionPath, meterPath, tempoPath)
            % [beats] = findBeatTimes(positionPath, meterPath, param_g)
            %   Find beat times from sequence of bar positions of the HMM beat tracker
            % ----------------------------------------------------------------------
            %INPUT parameter:
            % positionPath             : sequence of position states
            % meterPath                : sequence of meter states
            %                           NOTE: so far only median of sequence is used !
            % nBarPos                  : bar length in bar positions (=M)
            % framelength              : duration of being in one state in [sec]
            %
            %OUTPUT parameter:
            %
            % beats                    : [nBeats x 2] beat times in [sec] and
            %                           bar.beatnumber
            %
            % 29.7.2012 by Florian Krebs
            % ----------------------------------------------------------------------
            numframes = length(positionPath);
            meter = obj.meter_state2meter(:, meterPath);
            % TODO: if beat is shortly before the audio start we should
            % add one beat at the beginning. E.g. m-sequence starts with
            % m=2
            
            %             [meter_states, idx, ~] = unique(meterPath);
            for iT=1:size(obj.meter_state2meter, 2)
                beatpositions{iT} = round(linspace(1, obj.Meff(iT), obj.meter_state2meter(1, iT) + 1));
                beatpositions{iT} = beatpositions{iT}(1:end-1);
            end
            
            beats = [];
            beatno = [];
            beatco = 0;
            for i = 1:numframes-1
                for b = 1:length(beatpositions{meterPath(i)})
                    if positionPath(i) == beatpositions{meterPath(i)}(b)
                        beats = [beats; i];
                        beatno = [beatno; beatco + b/10];
                        if b == meter(1, i), beatco = beatco + 1; end
                        break;
                    elseif ((positionPath(i) > beatpositions{meterPath(i)}(b)) && (positionPath(i+1) > beatpositions{meterPath(i)}(b)) && (positionPath(i) > positionPath(i+1)))
                        % transition of two bars
                        %                         if positionPath(i+1) == mod(positionPath(i) + tempoPath(i) - 1, obj.Meff(meterPath(i))) + 1;
                        bt = interp1([positionPath(i); obj.M+positionPath(i+1)],[i; i+1],obj.M+beatpositions{meterPath(i)}(b));
                        beats = [beats; round(bt)];
                        beatno = [beatno; beatco + b/10];
                        if b == meter(1, i), beatco = beatco + 1; end
                        break;
                        %                         end
                    elseif ((positionPath(i) < beatpositions{meterPath(i)}(b)) && (positionPath(i+1) > beatpositions{meterPath(i)}(b)))
                        %                         if positionPath(i+1) == mod(positionPath(i) + tempoPath(i) - 1, obj.Meff(meterPath(i))) + 1;
                        bt = interp1([positionPath(i); positionPath(i+1)],[i; i+1],beatpositions{meterPath(i)}(b));
                        beats = [beats; round(bt)];
                        beatno = [beatno; beatco + b/10];
                        if b == meter(1, i), beatco = beatco + 1; end
                        break;
                        %                         end
                    end
                end
            end
            % if positionPath(i) == beatpositions(b), beats = [beats; i]; end
            
            beats = beats * obj.frame_length;
            beats = [beats beatno];
        end
        
    end
    
    methods (Static)
        
        [m, n] = getpath(M, annots, frame_length, nFrames);
        
    end
    
    
end
