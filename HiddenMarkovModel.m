classdef HiddenMarkovModel < handle
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        trans_model
        obs_model
        init_distribution
        n_states
    end
    
    methods
        function obj = HiddenMarkovModel(transition_model, observation_model, ...
                initial_distribution)
            obj.trans_model = transition_model;
            obj.obs_model = observation_model;
            obj.n_states = obj.trans_model.state_space.n_states;
            if exist('initial_distribution', 'var')
                obj.init_distribution = initial_distribution;
            else
                % assume uniform distribution
                obj.init_distribution = ones(obj.n_states, 1) / obj.n_states;
            end
            
        end
        
        function path = viterbi(obj, observations, use_mex_viterbi)
            if ~exist('use_mex_viterbi', 'var')
                use_mex_viterbi = 1;
            end
            % compute observation likelihoods
            obs_lik = obj.obs_model.compute_obs_lik(observations);
            if use_mex_viterbi
                try
                    path = obj.viterbi_mex(obs_lik);
                catch
                    fprintf(['\n    WARNING: viterbi.cpp has to be', ...
                        'compiled, using the pure MATLAB version', ...
                        'instead\n']);
                    path = obj.viterbi_matlab(obs_lik);
                end
            else
                path = obj.viterbi_matlab(obs_lik);
            end
        end
        
        function path = forward(obj, observations)
            
        end
    end
    
    
    methods (Access = protected)
        function bestpath = viterbi_matlab(obj, obs_lik)
            % [ bestpath, delta, loglik ] = viterbi_cont_int( A, obslik, y,
            % init_distribution)
            % Implementation of the Viterbi algorithm
            % ----------------------------------------------------------------------
            %INPUT parameter:
            % obj.trans_model.A     : transition matrix
            % obslik                : observation likelihood [R x nBarGridSize x nFrames]
            % obj.init_distribution      : initial state probabilities
            %
            %OUTPUT parameter:
            % bestpath      : MAP state sequence
            %
            % 26.7.2012 by Florian Krebs
            % ----------------------------------------------------------------------
            nFrames = size(obs_lik, 2);            
            % don't compute states that are irreachable:
            [row, col] = find(obj.trans_model.A);
            maxState = max([row; col]);
            minState = min([row; col]);
            nStates = maxState + 1 - minState;
            delta = obj.init_distribution;
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
                O = obs_lik(obj.obs_model.gmm_from_state(minState:maxState), ...
                    iFrame);
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
        
        function path = viterbi_mex(obj, obs_lik)
            
        end
        
    end
    
end

