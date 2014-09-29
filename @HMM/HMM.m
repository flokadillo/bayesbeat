classdef HMM
    % Hidden Markov Model Class
    properties
        M                   % number of (max) positions
        Meff                % number of positions per meter
        N                   % number of tempo states
        R                   % number of rhythmic pattern states
        pn                  % probability of a switch in tempo
        pr                  % probability of a switch in rhythmic pattern
        rhythm2meter_state  % assigns each rhythmic pattern to a meter state (1, 2, ...)
        meter_state2meter   % specifies meter for each meter state (9/8, 8/8, 4/4) [2 x nMeters]
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
        rhythm_names        % cell array of rhythmic pattern names
        train_dataset       % dataset, which HMM was trained on [string]
        correct_beats       % [0, 1, 2] correct beats after detection
        max_shift           % frame shifts that are allowed in forward path
        obs_lik_floor       % obs_lik has to be > floor to avoid overfitting
        update_interval     % best state sequence is set to the global max each update_interval
        use_silence_state
        pfs                 % transition from silence
        p2s                 % transition to silence
    end
    
    methods
        function obj = HMM(Params, meter_state2meter, rhythm2meter_state, rhythm_names)
            
            obj.M = Params.M;
            obj.N = Params.N;
            obj.R = Params.R;
            obj.pn = Params.pn;
            if isfield(Params, 'cluster_transitions_fln') && exist(Params.cluster_transitions_fln, 'file')
                obj.pr = dlmread(Params.cluster_transitions_fln);
            else
                obj.pr = Params.pr;
            end
            obj.barGrid = max(Params.whole_note_div * (meter_state2meter(1, :) ./ meter_state2meter(2, :)));
            obj.frame_length = Params.frame_length;
            obj.dist_type = Params.observationModelType;
            obj.init_n_gauss = Params.init_n_gauss;
            obj.rhythm2meter_state = rhythm2meter_state;
            obj.meter_state2meter = meter_state2meter;
            % effective number of bar positions per meter_state
            obj.Meff = round((meter_state2meter(1, :) ./ meter_state2meter(2, :)) ...
                * (Params.M ./ max(meter_state2meter(1, :) ./ meter_state2meter(2, :))));
            obj.pattern_size = Params.pattern_size;
            if isfield(Params, 'save_inference_data')
                obj.save_inference_data = Params.save_inference_data;
            else
                obj.save_inference_data = 0;
            end
            obj.tempo_tying = Params.tempo_tying;
            if isfield(Params, 'viterbi_learning_iterations')
                obj.viterbi_learning_iterations = Params.viterbi_learning_iterations;
            else
                obj.viterbi_learning_iterations = 0;
            end
            obj.n_depends_on_r = Params.n_depends_on_r;
            if isfield(Params, 'online')
                obj.max_shift = Params.online.max_shift;
                obj.update_interval = Params.online.update_interval;
                obj.obs_lik_floor = Params.online.obs_lik_floor;
            end
            obj.rhythm_names = rhythm_names;
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
        end
        
        function obj = make_transition_model(obj, minTempo, maxTempo)
            % convert from BPM into barpositions / audio frame
            meter_num = obj.meter_state2meter(1, obj.rhythm2meter_state);
%             meter_denom = obj.meter_state2meter(2, obj.rhythm2meter_state);
            
            if strcmp(obj.pattern_size, 'bar')
                obj.minN = floor(obj.Meff(obj.rhythm2meter_state) .* obj.frame_length .* minTempo ./ (meter_num * 60));
                obj.maxN = ceil(obj.Meff(obj.rhythm2meter_state) .* obj.frame_length .* maxTempo ./ (meter_num * 60));
            else
                obj.minN = floor(obj.M * obj.frame_length * minTempo ./ 60);
                obj.maxN = ceil(obj.M * obj.frame_length * maxTempo ./ 60);
            end
            
            if max(obj.maxN) ~= obj.N
                fprintf('    N should be %i instead of %i -> corrected\n', max(obj.maxN), obj.N);
                obj.N = max(obj.maxN);
            end
            
            if ~obj.n_depends_on_r % no dependency between n and r
                obj.minN = ones(1, obj.R) * min(obj.minN);
                obj.maxN = ones(1, obj.R) * max(obj.maxN);
                obj.N = max(obj.maxN);
                % fprintf('    Tempo limited to %i - %i bpm for all rhythmic patterns\n', round(min(obj.minN)*60*min(meter_denom)/(obj.M * obj.frame_length)), ...
                %    round(max(obj.maxN)*60*max(meter_denom)/(obj.M * obj.frame_length)));
                %                     obj.minN = ones(1, obj.R) * 8;
                %                     obj.maxN = ones(1, obj.R) * obj.N;
            end
            
            % Create transition model
            if obj.viterbi_learning_iterations > 0 % for viterbi learning use uniform tempo prior
                %    obj.minN = ones(1, obj.R);
                %    obj.maxN = ones(1, obj.R) * obj.N;
            end
            
            for r_i = 1:obj.R
                fprintf('    R=%i: Tempo limited to %i - %i bpm (%i - %i)\n', ...
                    r_i, round(obj.minN(r_i)*60*meter_num(r_i)/(obj.Meff(obj.rhythm2meter_state(r_i)) * obj.frame_length)), ...
                    round(obj.maxN(r_i)*60*meter_num(r_i)/(obj.Meff(obj.rhythm2meter_state(r_i)) * obj.frame_length)), ...
                    obj.minN(r_i), obj.maxN(r_i));
            end
            
            obj.trans_model = TransitionModel(obj.M, obj.Meff, obj.N, obj.R, obj.pn, obj.pr, ...
                obj.rhythm2meter_state, obj.minN, obj.maxN, obj.use_silence_state, obj.p2s, obj.pfs);
            
            % Check transition model
            if transition_model_is_corrupt(obj.trans_model, 0)
                error('Corrupt transition model');
            end
            
        end
        
        function obj = make_observation_model(obj, train_data)
            
            % Create observation model
            obj.obs_model = ObservationModel(obj.dist_type, obj.rhythm2meter_state, ...
                obj.meter_state2meter, obj.M, obj.N, obj.R, obj.barGrid, obj.Meff, train_data.feat_type, obj.use_silence_state);
            
            % Train model
            if obj.use_silence_state
                obj.obs_model = obj.obs_model.train_model(train_data);
            else
                obj.obs_model = obj.obs_model.train_model(train_data);
            end
            
            obj.train_dataset = train_data.dataset;
            
        end
        
        function obj = make_initial_distribution(obj, tempo_per_cluster)
            n_states = obj.M * obj.N * obj.R;
            
            if obj.use_silence_state
                % always start in the silence state
                obj.initial_prob = zeros(n_states+1, 1);
                obj.initial_prob(n_states+1) = 1;
            else
                if obj.init_n_gauss > 0
                    obj.initial_prob = zeros(n_states, 1);
                    for iCluster = 1:size(tempo_per_cluster, 2)
                        meter = obj.meter_state2meter(:, obj.rhythm2meter_state(iCluster));
                        tempi = tempo_per_cluster(:, iCluster) * obj.M * obj.frame_length ...
                            / (60 * meter(2));
                        gmm = gmdistribution.fit(tempi(~isnan(tempi)), obj.init_n_gauss);
                        % gmm_wide = gmdistribution(gmm.mu, gmm.Sigma, gmm.PComponents);
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
        
        function results = do_inference(obj, y, fname, inference_method, do_output)
            
            % compute observation likelihoods
            if strcmp(obj.dist_type, 'RNN')
                % normalize
                % TODO: norm to max=0.95 instead of 1
                for iR = 1:size(y, 2)
                    y(: ,iR) = y(: ,iR) / max(y(: ,iR));
                end
                obs_lik = obj.rnn_format_obs_prob(y);
                if obj.R  ~= size(y, 2)
                    error('Dim of RNN probs should be equal to R!\n');
                end
            else
                obs_lik = obj.obs_model.compute_obs_lik(y);
            end
            
            if strfind(inference_method, 'forward')
                % HMM forward path
                [hidden_state_sequence, alpha, psi, min_state] = obj.forward_path(obs_lik, do_output);
                [m_path, n_path, r_path] = ind2sub([obj.M, obj.N, obj.R], psi(:)');
                %                 [m_path, n_path, r_path] = obj.refine_forward_path(m_path, n_path, r_path, psi, min_state);
                %                 dlmwrite(['./data/filip/', fname, '-alpha.txt'], single(alpha(:, 1:200)));
                %                 dlmwrite(['./data/filip/', fname, '-best_states.txt'], uint32(psi(1:200)));
            elseif strfind(inference_method, 'viterbi')
                % decode MAP state sequence using Viterbi
                hidden_state_sequence = obj.viterbi_decode(obs_lik, fname);
                [m_path, n_path, r_path] = ind2sub([obj.M, obj.N, obj.R], hidden_state_sequence(:)');
            else
                error('inference method not specified\n');
            end
            
            %  figure;
            %  ax(1) = subplot(3, 1, 1);
            %  plot(m_path)
            %  ylabel('bar position')
            %  ax(2) = subplot(3, 1, 2);
            %  plot(r_path)
            %  ylabel('rhythm pattern')
            %  ax(3) = subplot(3, 1, 3);
            %  plot(y)
            %  ylabel('observation feature')
            %  linkaxes(ax,'x');
            
            
            t_path = zeros(length(r_path), 1);
            t_path(r_path<=obj.R) = obj.rhythm2meter_state(r_path(r_path<=obj.R));
            
            % compute beat times and bar positions of beats
            meter = zeros(2, length(r_path));
            meter(:, r_path<=obj.R) = obj.meter_state2meter(:, t_path(r_path<=obj.R));
            beats = obj.find_beat_times(m_path, t_path, n_path);
            
            %             anns=load(['~/diss/data/beats/smc_beats/annotations/beats/', fname]);
            %             figure; plot(y); hold on; stem(anns*100, ones(size(anns))*max(y(:)), 'r'); stem(beats(:, 1)*100, ones(size(beats(:, 1)))*max(y(:)), 'c--');
            
            if strcmp(obj.pattern_size, 'bar')
                tempo = meter(1, :) .* 60 .* n_path ./ (obj.Meff(obj.rhythm2meter_state(r_path)) * obj.frame_length);
            else
                tempo = 60 .* n_path / (obj.M * obj.frame_length);
                %                 obj.minN = floor(obj.M * obj.frame_length * minTempo ./ 60);
                %                 obj.maxN = ceil(obj.M * obj.frame_length * maxTempo ./ 60);
            end
            if obj.correct_beats > 0
                tempo_smooth = sgolayfilt(tempo, 1, min([211, length(tempo)]));
                win = 60 ./ (tempo_smooth(round(beats(:, 1)/obj.frame_length)) * meter(1, 1) * obj.barGrid);
            end
            if obj.correct_beats == 1
                % method 1
                beats(:, 1) = beats(:, 1) + win'./2;
            elseif obj.correct_beats == 2
                % method 2
                for i=1:length(beats(:, 1))
                    beat_frame = round(beats(i, 1)/obj.frame_length);
                    end_win_frame = min([beat_frame+round(win(i)/obj.frame_length),length(tempo)]);
                    [~, max_frame] = max(y(beat_frame:end_win_frame, t_path(1)));
                    beats(i, 1) = beats(i, 1) + (max_frame-1) * obj.frame_length;
                end
            end
            %             hold on; stem(beats(:, 1)*100, ones(size(beats(:, 1)))*max(y(:)), 'k');
            rhythm = r_path(r_path<=obj.R);
            results{1} = beats;
            results{2} = tempo;
            results{3} = meter;
            results{4} = r_path;
            results{5} = hidden_state_sequence;
        end
        
        function [obj, bar2cluster] = viterbi_training(obj, features, train_data)
            n_files = length(train_data.file_list);
            A_n = zeros(obj.N * obj.R, obj.N);
            A_r = zeros(obj.R, obj.R);
            observation_per_state = cell(n_files, obj.R, obj.barGrid, features.feat_dim);
            init = zeros(obj.N * obj.R, 1);
            % pattern id that each bar was assigned to in viterbi
            bar2cluster = zeros(size(train_data.bar2cluster));
            %    log_prob = zeros(n_files);
            
            for i_file=1:n_files
                % for i_file=326
                [~, fname, ~] = fileparts(train_data.file_list{i_file});
                fprintf('  %i/%i) %s', i_file, n_files, fname);
                % make belief function
                belief_func = obj.make_belief_functions(train_data, i_file);
                % load observations
                observations = features.load_feature(train_data.file_list{i_file});
                obs_lik = obj.obs_model.compute_obs_lik(observations);
                first_frame = max([belief_func{1}(1), 1]);
                %                 end_frame = min([belief_func{1}(end), size(obs_lik, 3)]); % make sure that belief_func is < nFrames
                % plot assignment from hiden states to observations before
                % training iteration
                %                 obj.visualise_hidden_states(observations(first_frame:end_frame, :), train_data.barpos_per_frame{i_file}(first_frame:end_frame), ...
                %                     train_data.pattern_per_frame{i_file}(first_frame:end_frame));
                
                best_path = obj.viterbi_iteration(obs_lik, belief_func);
                if isempty(best_path)
                    continue;
                end
                % plot assignment from hidden states to observations after
                % training iteration
                %                 obj.visualise_hidden_states(observations(first_frame:end_frame, :), best_path);
                %  log_prob(i_file) = compute_posterior(best_path, obs_lik, first_frame, obj);
                %                 fprintf('    log_prob=%.2f\n', log_prob(i_file));
                [m_path, n_path, r_path] = ind2sub([obj.M, obj.N, obj.R], best_path(:)');
                
                % %                 % compute beat times and bar positions of beats
                t_path = obj.rhythm2meter_state(r_path);
                beats = obj.find_beat_times(m_path, t_path, n_path);
                beats(:, 1) = beats(:, 1) + (belief_func{1}(1)-1) * obj.frame_length;
                %   BeatTracker.save_beats(beats, ['temp/', fname, '.beats.txt']);
                
                if min(n_path) < 5
                    fprintf('    Low tempo detected at file (n=%i), doing nothing\n', min(n_path));
                    %     continue;
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
                obj.minN(r_i) = find(sum(obj.trans_model.tempo_transition_probs((r_i-1)*obj.N + 1:r_i*obj.N, :), 2), 1, 'first');
                obj.maxN(r_i) = find(sum(obj.trans_model.tempo_transition_probs((r_i-1)*obj.N + 1:r_i*obj.N, :), 2), 1, 'last');
            end
            if ~obj.n_depends_on_r % no dependency between n and r
                obj.minN = ones(1, obj.R) * min(obj.minN);
                obj.maxN = ones(1, obj.R) * max(obj.maxN);
                fprintf('    New tempo limits: %i - %i\n', obj.minN(1), obj.maxN(1));
            end
            obj.trans_model = TransitionModel(obj.M, obj.Meff, obj.N, obj.R, obj.pn, obj.pr, ...
                obj.rhythm2meter_state, obj.minN, obj.maxN);
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
            %             if length(Params.feat_type) > 1
            %                 error('ERROR HMM.m: So far, only 1d features supported\n');
            %             end
            % save transition matrix
            transition_model = obj.trans_model.A;
            % save initial distribution
            initial_prob = obj.initial_prob;
            % save observation model
            observation_model = zeros(obj.R * obj.obs_model.barGrid, 6);
            for i_r=1:obj.R
                for i_pos=1:obj.obs_model.barGrid_eff(obj.obs_model.rhythm2meter_state(i_r))
                    observation_model(i_pos+(i_r-1)*obj.obs_model.barGrid, 1:2) = obj.obs_model.learned_params{i_r, i_pos}.mu;
                    observation_model(i_pos+(i_r-1)*obj.obs_model.barGrid, 3:4) = obj.obs_model.learned_params{i_r, i_pos}.Sigma;
                    observation_model(i_pos+(i_r-1)*obj.obs_model.barGrid, 5:6) = obj.obs_model.learned_params{i_r, i_pos}.PComponents;
                end
            end
            if obj.use_silence_state
                observation_model(obj.R * obj.obs_model.barGrid + 1, 1:2) = obj.obs_model.learned_params{obj.R+1, 1}.mu;
                observation_model(obj.R * obj.obs_model.barGrid + 1, 3:4) = obj.obs_model.learned_params{obj.R+1, 1}.Sigma;
                observation_model(obj.R * obj.obs_model.barGrid + 1, 5:6) = obj.obs_model.learned_params{obj.R+1, 1}.PComponents;
            end
            N = obj.N;
            M = obj.M;
            R = obj.R;
            P = obj.barGrid;
            
            % state 2 obs index
            state_to_obs = uint8(obj.obs_model.state2obs_idx);
            rhythm_to_meter = obj.meter_state2meter(:, obj.rhythm2meter_state);
            
            tempo_ranges = zeros(2, obj.R);
            tempo_ranges(1, :) = obj.minN .* rhythm_to_meter(2, :) * 60 / (obj.M * obj.frame_length);
            tempo_ranges(2, :) = obj.maxN .* rhythm_to_meter(2, :) * 60 / (obj.M * obj.frame_length);
            
            %save to mat file
            save(fullfile(folder, 'robot_hmm_data.mat'), 'M', 'N', 'R', 'P' ,'transition_model', ...
                'observation_model', 'initial_prob', 'state_to_obs', 'rhythm_to_meter', 'tempo_ranges', '-v7.3');
            fprintf('* Saved model data (Flower) to %s\n', fullfile(folder, 'robot_hmm_data.mat'));
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
                %     fprintf('    Size of Psi = %.1f MB\n', nStates * nFrames * 4 / 10^6);
                psi_mat = zeros(nStates, nFrames, 'uint32');  % 32 bit unsigned integer
            else
                %     fprintf('    Size of Psi = %.1f MB\n', nStates * nFrames * 2 / 10^6);
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
            validInds = ~isnan(ind); %
            O(validInds) = obs_lik(ind(validInds));
            O(validInds & (O<1e-10)) = 1e-10;
            delta = O .* delta;
            delta = delta / sum(delta);
            %             delta = zeros(size(delta));
            %             delta(14000) = 1;
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
                %                 O = zeros(nStates, 1);
                %                 validInds = ~isnan(ind);
                %                 sum(validInds)
                % ind is shifted at each time frame -> all frames are used
                O(validInds) = obs_lik(ind(validInds));
                O(validInds & (O<1e-10)) = 1e-10;
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
                %                 if sum(belief_func{1} == iFrame+start_frame-1)
                if ismember(iFrame+start_frame-1, belief_func{1})
                    delta_max = delta_max .* belief_func{2}(belief_func{1} == iFrame+start_frame-1, minState:maxState);
                    delta_max = delta_max / sum(delta_max);
                    if sum(isnan(delta_max)) > 0
                        fprintf(' Viterbi path could not be determined (error at frame %i)\n', iFrame);
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
            bestpath = zeros(nFrames, 1);
            [ ~, bestpath(nFrames)] = max(delta);
            for iFrame=nFrames-1:-1:1
                bestpath(iFrame) = psi_mat(bestpath(iFrame+1),iFrame+1);
            end
            
            % add state offset
            bestpath = bestpath + minState - 1;
            fprintf(' done\n');
        end
        
        function [marginal_best_bath, alpha, best_states, minState] = forward_path(obj, obs_lik, do_output)
            % HMM forward path
            store_alpha = 0;
            
            nFrames = size(obs_lik, 3);
            
            % don't compute states that are irreachable:
            [row, col] = find(obj.trans_model.A);
            maxState = max([row; col]);
            minState = min([row; col]);
            nStates = maxState + 1 - minState;
            
            A = obj.trans_model.A(minState:maxState, minState:maxState);
            %             i_row = 1:nStates;
            %             j_col = 1:nStates;
            
            %             psi = zeros(nStates, nFrames, 'uint16'); % 16 bit unsigned integer
            
            if store_alpha
                alpha = zeros(nStates, nFrames);
                alpha(:, 1) = obj.initial_prob(minState:maxState);
                alpha(:, 1) = A' * alpha(:, 1); % first transition ftom t=0 to t=1
            else
                alpha = obj.initial_prob(minState:maxState);
                alpha = A' * alpha;
            end
            
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
            %             O(validInds & (O<1e-3)) = 1e-3;
            if store_alpha
                alpha(:, 1) = O .* alpha(:, 1);
                alpha(:, 1) = alpha(:, 1) / sum(alpha(:, 1));
                [~, best_states(1)] = max(alpha(:, 1));
            else
                alpha = O .* alpha;
                alpha = alpha / sum(alpha);
                [~, best_states(1)] = max(alpha);
            end
            % move pointer to next observation
            ind = ind + ind_stepsize;
            if do_output, fprintf('    Forward path .'); end
            for iFrame = 2:nFrames
                if store_alpha
                    alpha(:, iFrame) = A' * alpha(:, iFrame-1);
                else
                    alpha = A' * alpha;
                end
                %                 D = sparse(i_row, j_col, alpha(:, iFrame), nStates, nStates);
                %                 [ ~, psi(:, iFrame)] = max(bsxfun(@times, A, alpha(:, iFrame-1)));
                % compute likelihood p(yt|x1:t)
                %                 O = zeros(nStates, 1);
                %                 validInds = ~isnan(ind);
                % ind is shifted at each time frame -> all frames are used
                O(validInds) = obs_lik(ind(validInds));
                O(validInds & (O < obj.obs_lik_floor)) = obj.obs_lik_floor;
                % increase index to new time frame
                ind = ind + ind_stepsize;
                if store_alpha
                    alpha(:, iFrame) = O .* alpha(:, iFrame);
                    % normalize0
                    norm_const = sum(alpha(:, iFrame));
                    alpha(:, iFrame) = alpha(:, iFrame) / norm_const;
                else
                    alpha = O .* alpha;
                    % normalize0
                    norm_const = sum(alpha);
                    alpha = alpha / norm_const;
                end
                if rem(iFrame, perc) == 0 && do_output
                    fprintf('.');
                end
                
                if rem(iFrame, obj.update_interval) == 0
                    % use global maximum as best state
                    if store_alpha
                        [~, best_states(iFrame)] = max(alpha(:, iFrame));
                    else
                        [~, best_states(iFrame)] = max(alpha);
                    end
                else
                    C = A(best_states(iFrame-1), :)' .* O;
                    if nnz(C) == 0
                        if store_alpha
                            [~, best_states(iFrame)] = max(alpha(:, iFrame));
                        else
                            [~, best_states(iFrame)] = max(alpha);
                        end
                    else
                        % find best state among a restricted set of
                        % possible successor states
                        possible_successors = find(A(best_states(iFrame-1), :)) + minState - 1;
                        [m, n, r] = ind2sub([obj.M, obj.N, obj.R], possible_successors);
                        m_extended = [];
                        n_extended = [];
                        r_extended = [];
                        for i_s = find(r<=obj.R) % loop over all possible successors
                            % allow position shift for each possible
                            % successor
                            m_extended = [m_extended, m(i_s):-1:(m(i_s)-obj.max_shift), m(i_s)+1:+1:(m(i_s)+obj.max_shift)];
                            m_extended = mod(m_extended - 1, obj.Meff(obj.rhythm2meter_state(r(i_s)))) + 1; % new position
                            n_extended = [n_extended, ones(1, 2*obj.max_shift+1)*n(i_s)];
                            r_extended = [r_extended, ones(1, 2*obj.max_shift+1)*r(i_s)];
                        end
                        possible_successors = sub2ind([obj.M, obj.N, obj.R], m_extended, ...
                            n_extended, r_extended);
                        if sum(r>obj.R)>0
                            possible_successors = [possible_successors, sub2ind([obj.M, obj.N, obj.R+1], 1, 1, obj.R+1)];
                        end
                        possible_successors = possible_successors - minState + 1;
                        if store_alpha
                            [~, idx] = max(alpha(possible_successors, iFrame));
                        else
                            [~, idx] = max(alpha(possible_successors));
                        end
                        best_states(iFrame) = possible_successors(idx);
                    end
                end
            end
            if store_alpha
                [~, marginal_best_bath] = max(alpha);
            else
                marginal_best_bath = [];
            end
            % add state offset
            marginal_best_bath = marginal_best_bath + minState - 1;
            best_states = best_states + minState - 1;
            if do_output, fprintf(' done\n'); end
        end
        
        function [m_path_new, n_path_new, r_path_new] = refine_forward_path1(obj, m_path, n_path, r_path)
            % Wait until Kalman filter sets in
            wait_sec = 3;
            
            addpath('~/diss/src/matlab/libs/matlab_utils')
            m_path_new = m_path;
            n_path_new = n_path;
            r_path_new = r_path;
            
            wait_int = min([max([round(wait_sec / obj.frame_length), 1]), length(m_path)-1]);
            c=0;
            par.A = [1, 1; 0, 1];
            par.Q = [0.1, 0; 0, 0.001];
            par.C = [1, 0];
            par.R = 500;
            P = ones(2, 2);
            x = [m_path_new(wait_int); n_path_new(wait_int)];
            for iFrame = wait_int+1:length(m_path)
                if abs(m_path(iFrame) - x(1)) < abs(m_path(iFrame) + obj.Meff(r_path(iFrame))- x(1))
                    y = m_path(iFrame);
                else
                    y = m_path(iFrame) + obj.Meff(r_path(iFrame));
                end
                x_old = x;
                [ x, P, ~, ~, ~, ~ ] = KF( x, P, y, par);
                m_path_new(iFrame) = mod(x(1) - 1, obj.Meff(r_path(iFrame))) + 1;
                x(1) = mod(x(1) - 1, obj.Meff(r_path(iFrame))) + 1;
                n_path_new(iFrame) = x(2);
            end
            %             figure(1); plot(m_path(1:iFrame)); hold on; plot(m_path_new(1:iFrame), '--r')
        end
        
        function [m_path_new, n_path_new, r_path_new] = refine_forward_path2(obj, m_path, n_path, r_path)
            % Wait until Kalman filter sets in
            update_sec = 3;
            
            addpath('~/diss/src/matlab/libs/matlab_utils')
            m_path_new = m_path;
            n_path_new = n_path;
            r_path_new = r_path;
            
            update_int = min([max([round(update_sec / obj.frame_length), 1]), length(m_path)-1]);
            [ ~, best_state(1)] = max(alpha(:, 1));
            for iFrame = 1:length(m_path)
                
                if rem(iFrame, update_int) == 0
                    [ ~, best_state(iFrame)] = max(alpha);
                else
                    best_state(iFrame) = psi(best_state(iFrame))
                end
                
            end
            %             figure(1); plot(m_path(1:iFrame)); hold on; plot(m_path_new(1:iFrame), '--r')
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
            meter = zeros(2, length(meterPath));
            meter(:, meterPath>0) = obj.meter_state2meter(:, meterPath(meterPath>0));
            % TODO: if beat is shortly before the audio start we should
            % add one beat at the beginning. E.g. m-sequence starts with
            % m=2
            for iT=1:size(obj.meter_state2meter, 2)
                beatpositions{iT} = round(linspace(1, obj.Meff(iT), obj.meter_state2meter(1, iT) + 1));
                beatpositions{iT} = beatpositions{iT}(1:end-1);
            end
            
            beatno = [];
            bar_number = 0;
            beats = [];
            for i = 1:numframes-1
                if meterPath(i) == 0
                    continue;
                end
                for beat_pos = 1:length(beatpositions{meterPath(i)})
                    if positionPath(i) == beatpositions{meterPath(i)}(beat_pos)
                        % current frame = beat frame
                        beats = [beats; [i, bar_number, beat_pos]];
                        if beat_pos == meter(1, i), bar_number = bar_number + 1; end
                        break;
                    elseif ((positionPath(i+1) > beatpositions{meterPath(i)}(beat_pos)) && (positionPath(i+1) < positionPath(i)))
                        % bar transition between frame i and frame i+1
                        bt = interp1([positionPath(i); obj.M+positionPath(i+1)],[i; i+1],obj.M+beatpositions{meterPath(i)}(beat_pos));
                        beats = [beats; [round(bt), bar_number, beat_pos]];
                        if beat_pos == meter(1, i), bar_number = bar_number + 1; end
                        break;
                    elseif ((positionPath(i) < beatpositions{meterPath(i)}(beat_pos)) && (beatpositions{meterPath(i)}(beat_pos) < positionPath(i+1)))
                        % beat position lies between frame i and frame i+1
                        bt = interp1([positionPath(i); positionPath(i+1)],[i; i+1],beatpositions{meterPath(i)}(beat_pos));
                        beats = [beats; [round(bt), bar_number, beat_pos]];
                        if beat_pos == meter(1, i), bar_number = bar_number + 1; end
                        break;
                    end
                end
            end
            % if positionPath(i) == beatpositions(b), beats = [beats; i]; end
            
            beats(:, 1) = beats(:, 1) * obj.frame_length;
        end
        
        function belief_func = make_belief_functions(obj, train_data, file_ids)
            if nargin < 3
                % compute belief functions for the whole dataset
                file_ids = 1:length(train_data.file_list);
            end
            tol_beats = 0.0875; % tolerance window in percentage of one beat period
            method_type = 2;
            tol_bpm = 20; % tolerance in +/- bpm for tempo variable
            % compute tol_win in [frames]
            %             tol_win_m = floor(tol_beats * obj.Meff(1) / obj.meter_state2meter(1, 1));
            %             tol_win_n = floor(obj.Meff(1) .* obj.frame_length * tol_bpm ./ (obj.meter_state2meter(1, 1) .* 60));
            % belief_func:
            % col1: frames where annotation is available,
            % col2: sparse vector that is one for possible states
            belief_func = cell(length(file_ids), 2);
            n_states = obj.M * obj.N * obj.R;
            counter = 1;
            for i_file = file_ids(:)'
                % find rhythm of current piece (so far, only one (the first) per piece is used!)
                %                 rhythm_id = train_data.bar2cluster(find(train_data.bar2file==i_file, 1));
                
                % find meter of current piece (so far, only one (the first) per piece is used!)
                if train_data.meter(i_file) == 0 % no meter annotation available
                    possible_meter_states = 1:size(obj.meter_state2meter, 2); % all meters are possible
                else
                    possible_meter_states = find((obj.meter_state2meter(1, :) == train_data.meter(i_file, 1)) &...
                        (obj.meter_state2meter(2, :) == train_data.meter(i_file, 2)));
                end
                % compute inter beat intervals in seconds
                ibi = diff(train_data.beats{i_file}(:, 1));
                ibi = [ibi; ibi(end)];
                % number of beats
                n_beats_i = size(train_data.beats{i_file}, 1);
                % estimate roughly the size of the tolerance windows to
                % pre-allocate memory (assuming 4/4 meter)
                tol_win_m = floor(tol_beats * obj.M / 4);
                %                 tol_win_n = floor(obj.M .* obj.frame_length * tol_bpm ./ (4 .* 60));
                % pre-allocate memory for rows, cols and values
                i_rows = zeros((tol_win_m*2+1) * n_beats_i * obj.N * obj.R * sum(obj.meter_state2meter(1, :)), 1);
                j_cols = zeros((tol_win_m*2+1) * n_beats_i * obj.N * obj.R * sum(obj.meter_state2meter(1, :)), 1);
                s_vals = ones((tol_win_m*2+1) * n_beats_i * obj.N * obj.R * sum(obj.meter_state2meter(1, :)), 1);
                p=1;
                for iMeter=possible_meter_states
                    % check if downbeat is annotated
                    if size(train_data.beats{i_file}, 2) == 1 % no downbeat annotation available
                        possible_btypes = repmat(1:obj.meter_state2meter(1, iMeter), n_beats_i, 1);
                    else
                        possible_btypes = train_data.beats{i_file}(:, 3); % position of beat in a bar: 1, 2, 3, 4
                    end
                    % find all rhythm states that belong to iMeter
                    r_state = find(obj.rhythm2meter_state == iMeter);
                    % convert each beat_type into a bar position {1..Meff}
                    M_i = obj.Meff(iMeter);
                    tol_win = floor(tol_beats * obj.M / train_data.meter(i_file, 2));
                    % compute bar position m for each beat. beats_m is
                    % [n_beats x 1] if downbeat is annotated or [n_beats x num(iMeter)]
                    % otherwise
                    beats_m = ((possible_btypes-1) .* M_i ./ obj.meter_state2meter(1, iMeter)) + 1;
                    for iM_beats=beats_m % loop over beat types
                        for iBeat=1:n_beats_i % loop over beats of file i
                            if method_type == 1
                                %                             % -----------------------------------------------------
                                %                             % Variant 1: tolerance win constant in beats over tempi
                                %                             % -----------------------------------------------------
                                m_support = mod((iM_beats(iBeat)-tol_win:iM_beats(iBeat)+tol_win) - 1, M_i) + 1;
                                m = repmat(m_support, 1, obj.N * length(r_state));
                                n = repmat(1:obj.N, length(r_state) * length(m_support), 1);
                                r = repmat(r_state(:), obj.N * length(m_support), 1);
                                states = sub2ind([obj.M, obj.N, obj.R], m(:), n(:), r(:));
                                idx = (iBeat-1)*(tol_win*2+1)*obj.N*length(r_state)+1:(iBeat)*(tol_win*2+1)*obj.N*length(r_state);
                                i_rows(idx) = iBeat;
                                j_cols(idx) = states;
                            elseif method_type == 2
                                % -----------------------------------------------------
                                % Variant 2: Tolerance win constant in time over tempi
                                % -----------------------------------------------------
                                
                                for n_i = obj.minN(r_state):obj.maxN(r_state)
                                    tol_win = n_i * tol_beats * ibi(iBeat) / obj.frame_length;
                                    m_support = mod((iM_beats(iBeat)-ceil(tol_win):iM_beats(iBeat)+ceil(tol_win)) - 1, M_i) + 1;
                                    for r_i = 1:length(r_state)
                                        states = sub2ind([obj.M, obj.N, obj.R], m_support(:), ones(size(m_support(:)))*n_i, ones(size(m_support(:)))*r_state(r_i));
                                        j_cols(p:p+length(states)-1) = states;
                                        i_rows(p:p+length(states)-1) = iBeat;
                                        p = p + length(states);
                                    end
                                end
                            end
                            %                             % tempo
                            %                             n_valid = (ibi(iBeat)-tol_win_n):(ibi(iBeat)+tol_win_n);
                            %                             n_valid(n_valid>obj.N) = obj.N;
                            %                             m_support = mod((iM_beats(iBeat)-tol_win_m:iM_beats(iBeat)+tol_win_m) - 1, M_i) + 1;
                            %                             m = repmat(m_support, 1, length(n_valid) * length(r_states));
                            %                             n = repmat(n_valid, length(r_states) * length(m_support), 1);
                            %                             r = repmat(r_states(:), length(n_valid) * length(m_support), 1);
                            %                             states = sub2ind([obj.M, obj.N, obj.R], m(:), n(:), r(:));
                            % %                             idx = (iBeat-1)*(tol_win_m*2+1)*obj.N*length(r_states)+1:(iBeat)*(tol_win_m*2+1)*obj.N*length(r_states);
                            %                             i_rows(p:p+length(m)-1) = iBeat;
                            %                             j_cols(p:p+length(m)-1) = states;
                            %                             p = p + length(m);
                        end
                    end
                end
                %                 [~, idx, ~] = unique([i_rows, j_cols], 'rows');
                belief_func{counter, 1} = round(train_data.beats{i_file}(:, 1) / obj.frame_length);
                belief_func{counter, 1}(1) = max([belief_func{counter, 1}(1), 1]);
                belief_func{counter, 2} = logical(sparse(i_rows(i_rows>0), j_cols(i_rows>0), s_vals(i_rows>0), n_beats_i, n_states));
                counter = counter + 1;
            end
        end
        
        function obs_lik = rnn_format_obs_prob(obj, y)
            obs_lik = zeros(size(y, 2), obj.barGrid, size(y, 1));
            for iR = 1:size(y, 2)
                obs_lik(iR, 1, :) = y(:, iR);
                obs_lik(iR, 2:end, :) = repmat((1-y(:, iR))/(obj.barGrid-1), 1, obj.barGrid-1)';
                %                 obs_lik(iR, 2:end, :) = repmat((1-y(:, iR)), 1, obj.barGrid-1)';
            end
            
        end
        
    end
    
    methods (Static)
        
        [m, n] = getpath(M, annots, frame_length, nFrames);
        
    end
    
    
end
