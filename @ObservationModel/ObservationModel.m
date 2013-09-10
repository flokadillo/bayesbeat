classdef ObservationModel
    % Observation Model Class
    properties (SetAccess=private)
        M                   % number of positions in a 4/4 bar
        N                   % number of tempo states
        R                   % number of rhythmic pattern states
        rhythm2meter        % assigns each rhythmic pattern to a meter
        barGrid             % number of different observation model params per bar (e.g., 64)
        dist_type           % type of parametric distribution
        obs_prob_fun_handle %
        learned_params      % cell array of learned parameters [nPatterns x nBarPos] 
        learned_params_all  % same as learned parameters but for all files (this is useful for 
                            % leave-one-out validation; in this case learned_params contains 
                            % parameters for all except one files)
        lik_func_handle     % function handle to likelihood function
        state2obs_idx       % specifies which states are tied (share the same parameters)
    end
    
    methods
        function obj = ObservationModel(dist_type, rhythm2meter, M, N, R, barGrid)
            obj.rhythm2meter = rhythm2meter;
            obj.dist_type = dist_type;
            obj.lik_func_handle = set_lik_func_handle(obj);
            obj.M = M;
            obj.N = N;
            obj.R = R;
            obj.barGrid = barGrid;
            fprintf('* Set up observation model .');
            obj = obj.make_state2obs_idx();
        end
                      
        params = fit_distribution(obj, data_file_pattern_barpos_dim)
         
        function obj = train_model(obj, data_file_pattern_barpos_dim)
            obj.learned_params = obj.fit_distribution(data_file_pattern_barpos_dim);
            % store learned params in case of leave-one-out testing, where
            % we update learned_params in each step
            obj.learned_params_all = obj.learned_params;
        end
        
        function obj = retrain_model(obj, data_file_pattern_barpos_dim, pattern_id)
            % restore learned_params to params trained on all files
            obj.learned_params = obj.learned_params_all;
            % update parameters of pattern_id
            obj.learned_params(pattern_id, :) = obj.fit_distribution(data_file_pattern_barpos_dim(:, pattern_id, :, :));
        end
        
        function obsLik = compute_obs_lik(obj, observations)
            % [ obsLik ] = compute_obs_lik(o1, observations)
            %
            % pre-computes the likelihood of the observations using the model o1
            %
            % ------------------------------------------------------------------------
            %INPUT parameters:
            % o1            : parameter of observation model
            % observations  : observations
            %
            %OUTPUT parameters:
            % obslik        : observation likelihood [R x barPos x nFrames]
            %
            % 21.03.2013 by Florian Krebs
            % ------------------------------------------------------------------------
            nFrames = size(observations, 1);
            obsLik = zeros(obj.R, obj.barGrid, nFrames);
            for iR = 1:obj.R
                barPos = obj.barGrid * (obj.rhythm2meter(iR) + 2)/4;
                obsLik(iR, 1:barPos, :) = obj.lik_func_handle(observations, ...
                    obj.learned_params(iR, 1:barPos));
            end
        end
        
        lik_func_handle = set_lik_func_handle(obj)
    end
    
    methods (Access=protected)
        
       function obj = make_state2obs_idx(obj)
            %{
            Computes state2obs_idx, which specifies which states are tied (share the same parameters) 
            %}
           
            nStates = obj.M * obj.N * obj.R;
            obj.state2obs_idx = nan(nStates, 2);
            barPosPerGrid = obj.M / obj.barGrid;
            discreteBarPos = floor(((1:obj.M) - 1)/barPosPerGrid) + 1;
            for iR=1:obj.R
                Meff = obj.M * (obj.rhythm2meter(iR)+2) / 4;
                r = ones(Meff, 1) * iR;
                for iN = 1:obj.N
                    ind = sub2ind([obj.M, obj.N, obj.R], (1:Meff)', repmat(iN, Meff, 1), r);
                    obj.state2obs_idx(ind, 1) = r;
                    obj.state2obs_idx(ind, 2) = discreteBarPos(1:Meff);
                end
            end
        end 
    end
end