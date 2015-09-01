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
            n_frames = size(obs_lik, 2);            
            delta = obj.init_distribution;
            A = obj.trans_model.A;
            if length(delta) > 65535
                % store psi as 32 bit unsigned integer
                psi_mat = zeros(obj.n_states, n_frames, 'uint32');  
            else
                % store psi as 16 bit unsigned integer
                psi_mat = zeros(obj.n_states, n_frames, 'uint16');
            end
            perc = round(0.1*n_frames);
            i_row = 1:obj.n_states;
            j_col = 1:obj.n_states;
            for i_frame = 1:n_frames
                % create a matrix that has the same value of delta for all 
                % entries with the same state i (row)
                % (same as repmat(delta, 1, col), but faster)
                D = sparse(i_row, j_col, delta(:), obj.n_states, obj.n_states);
                [delta, psi_mat(:, i_frame)] = max(D * A);
                % compute likelihood p(yt|x1:t)
                delta = obs_lik(obj.obs_model.gmm_from_state, i_frame) .* delta';
                % normalize
                delta = delta / sum(delta);
                if rem(i_frame, perc) == 0
                    fprintf('.');
                end
            end
            % Backtracing
            bestpath = zeros(n_frames, 1);
            [m, bestpath(n_frames)] = max(delta);
            maxIndex = find(delta == m);
            bestpath(n_frames) = round(median(maxIndex));
            for i_frame=n_frames-1:-1:1
                bestpath(i_frame) = psi_mat(bestpath(i_frame+1), i_frame+1);
            end
            fprintf(' done\n');
        end
        
        function path = viterbi_mex(obj, obs_lik)
            % convert transition matrix to three vectors containing the
            % from_state, to_state and the transition probability
            [state_ids_i, state_ids_j, trans_prob_ij] = find(obj.trans_model.A);
            path = obj.viterbi(state_ids_i, state_ids_j, trans_prob_ij, ...
                obj.initial_prob, obs_lik, obj.obs_model.gmm_from_state);
            fprintf(' done\n');
        end
        
    end
    
    methods (Static)
        [path] = viterbi(state_ids_i, state_ids_j, trans_prob_ij, ...
                initial_prob, obs_lik, gmm_from_state);
    end
    
end

