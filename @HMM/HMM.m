classdef HMM
    % Hidden Markov Model Class
    properties (SetAccess=private)
        M                   % number of (max) positions
        Meff                % number of positions per meter
        N                   % number of tempo states
        R                   % number of rhythmic pattern states
        pn                  % probability of a switch in tempo
        pr                  % probability of a switch in rhythmic pattern
        pt                  % probability of a switch in meter
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
        
    end
    
    methods
        function obj = HMM(Params, rhythm2meter)
            
            obj.M = Params.M;
            obj.Meff = Params.Meff;
            obj.N = Params.N;
            obj.R = Params.R;
            obj.pn = Params.pn;
            obj.pr = Params.pr;
            obj.pt = Params.pt;
            obj.barGrid = Params.barGrid;
            obj.frame_length = Params.frame_length;
            obj.dist_type = Params.observationModelType;
            obj.init_n_gauss = Params.init_n_gauss;
            obj.rhythm2meter = rhythm2meter;
            obj.meter_state2meter = Params.meters;
            
        end
        
        function obj = make_transition_model(obj, minTempo, maxTempo)
            % convert from BPM into barpositions / audio frame
            meter_denom = obj.meter_state2meter(2, :);
            meter_denom = meter_denom(obj.rhythm2meter);
            obj.minN = round(obj.M * obj.frame_length * minTempo ./ (meter_denom * 60));
            obj.maxN = round(obj.M * obj.frame_length * maxTempo ./ (meter_denom * 60));
            
            % Create transition model
            obj.trans_model = TransitionModel(obj.M, obj.Meff, obj.N, obj.R, obj.pn, obj.pr, ...
                obj.pt, obj.rhythm2meter, obj.minN, obj.maxN);
            
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
        
        function obj = make_initial_distribution(obj, use_tempo_prior, tempo_per_cluster)
            n_states = obj.M * obj.N * obj.R;
            if use_tempo_prior
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
            % decode MAP state sequence using Viterbi
            hidden_state_sequence = obj.viterbi_decode(obs_lik, fname);
            % factorial HMM: mega states -> substates
            [m_path, n_path, r_path] = ind2sub([obj.M, obj.N, obj.R], hidden_state_sequence);
            % meter path
            t_path = obj.rhythm2meter(r_path);
            % compute beat times and bar positions of beats
            meter = obj.meter_state2meter(:, t_path);
            beats = obj.find_beat_times(m_path, t_path);
            tempo = meter(2, :)' .* 60 .* n_path / (obj.M * obj.frame_length);
            rhythm = r_path;
        end
        
        
    end
    
    methods (Access=protected)
        
        
        function bestpath = viterbi_decode(obj, obs_lik, fname)
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
            save_data = 1;
            
            nFrames = size(obs_lik, 3);
            loglik = zeros(nFrames, 1);
            [row, col] = find(obj.trans_model.A);
            if save_data, logP_data = sparse(size(obj.trans_model.A, 1), nFrames); end
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
            
%             alpha = zeros(nFrames, 1); % most probable state for each frame given by forward path
            perc = round(0.1*nFrames);
            
            i_row = 1:nStates;
            j_col = 1:nStates;
            
            ind = sub2ind([obj.R, obj.barGrid, nFrames ], obj.obs_model.state2obs_idx(minState:maxState, 1), ...
                obj.obs_model.state2obs_idx(minState:maxState, 2), ones(nStates, 1));
            ind_stepsize = obj.barGrid * obj.R;
            
            fprintf('    Decoding (viterbi) .');
            
            for iFrame = 1:nFrames
                if save_data,
                   p_ind = find(log(delta) > -20);
                   logP_data(p_ind - 1 + minState, iFrame) = log(delta(p_ind));
                end
                % delta = prob of the best sequence ending in state j at time t, when observing y(1:t)
                % D = matrix of probabilities of best sequences with state i at time
                % t-1 and state j at time t, when bserving y(1:t)
                
                % create a matrix that has the same value of delta for all entries with
                % the same state i (row)
                % same as repmat(delta, 1, col)
                D = sparse(i_row, j_col, delta(:), nStates, nStates);
                [delta_max, psi_mat(:,iFrame)] = max(D * A);
                % compute likelihood p(yt|x1:t)
                O = zeros(nStates, 1);
                validInds = ~isnan(ind);
                O(validInds) = obs_lik(ind(validInds));
                % increase index to new time frame
                ind = ind + ind_stepsize;
                delta_max = O .* delta_max';
                % normalize
                norm_const = sum(delta_max);
                delta = delta_max / norm_const;
%                 [~, alpha(iFrame)] = max(delta);
                loglik(iFrame) = log(norm_const);
                if rem(iFrame, perc) == 0
                    fprintf('.');
                end
            end
            
            if save_data,
                % save for visualization
                M = obj.M; N = obj.N; R = obj.R; frame_length = obj.frame_length;
                save(['~/diss/src/matlab/beat_tracking/bayes_beat/temp/', fname, '_hmm.mat'], ...
                    'logP_data', 'M', 'N', 'R', 'frame_length', 'obs_lik');
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
        
        function beats = find_beat_times(obj, positionPath, meterPath)
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
            % TODO: implement for changes in meter
            if round(median(meter(1, :))) == 3 % 3/4
                numbeats = 3;
                denom = 4;
            elseif round(median(meter(1, :))) == 4 % 4/4
                numbeats = 4;
                denom = 4;
            elseif round(median(meter(1, :))) == 8 % 8/8
                numbeats = 8;
                denom = 8;
            elseif round(median(meter(1, :))) == 9 % 9/8
                numbeats = 9;
                denom = 8;
            else
                error('Meter %i not supported yet!\n', median(meterPath));
            end
            
            beatpositions =  round(linspace(1, obj.Meff(median(meterPath)), numbeats+1));
            beatpositions = beatpositions(1:end-1);
%             beatpositions = [1; round(obj.M/4)+1; round(obj.M/2)+1; round(3*obj.M/4)+1];
            
            beats = [];
            beatno = [];
            beatco = 0;
            for i = 1:numframes-1
                for b = 1:numbeats
                    if positionPath(i) == beatpositions(b)
                        beats = [beats; i];
                        beatno = [beatno; beatco + b/10];
                        if b == numbeats, beatco = beatco + 1; end
                    elseif ((positionPath(i) > beatpositions(b)) && (positionPath(i+1) > beatpositions(b)) && (positionPath(i) > positionPath(i+1)))
                        % transition of two bars
                        bt = interp1([positionPath(i); obj.M+positionPath(i+1)],[i; i+1],obj.M+beatpositions(b));
                        beats = [beats; round(bt)];
                        beatno = [beatno; beatco + b/10];
                        if b == numbeats, beatco = beatco + 1; end
                    elseif ((positionPath(i) < beatpositions(b)) && (positionPath(i+1) > beatpositions(b)))
                        bt = interp1([positionPath(i); positionPath(i+1)],[i; i+1],beatpositions(b));
                        beats = [beats; round(bt)];
                        beatno = [beatno; beatco + b/10];
                        if b == numbeats, beatco = beatco + 1; end
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
