classdef ObservationModel
    % Observation Model Class
    properties (SetAccess=private)
        M                   % max number of positions in a bar
        Meff                % number of positions per bar
        N                   % number of tempo states
        R                   % number of rhythmic pattern states
        rhythm2meter_state        % assigns each rhythmic pattern to a meter
        meter_state2meter   % specifies meter for each meter state (9/8, 8/8, 4/4)
        barGrid             % number of different observation model params per bar (e.g., 64)
        barGrid_eff         % number of distributions to fit per meter
        dist_type           % type of parametric distribution
        obs_prob_fun_handle %
        learned_params      % cell array of learned parameters [nPatterns x nBarPos] 
        learned_params_all  % same as learned parameters but for all files (this is useful for 
                            % leave-one-out validation; in this case learned_params contains 
                            % parameters for all except one files)
        lik_func_handle     % function handle to likelihood function
        state2obs_idx       % specifies which states are tied (share the same parameters)
                            % [nStates, 2]. first columns is the rhythmic
                            % pattern indicator, second one the bar
                            % position (e.g., 1, 2 .. 64 )
        use_silence_state
        feat_type           % cell array (features (extension) to be used)
    end
    
    methods
        function obj = ObservationModel(dist_type, rhythm2meter_state, ...
                meter_state2meter, M, N, R, barGrid, Meff, feat_type, ...
                use_silence_state)
            obj.rhythm2meter_state = rhythm2meter_state;
            obj.meter_state2meter = meter_state2meter;
            obj.dist_type = dist_type;
            obj.lik_func_handle = set_lik_func_handle(obj);
            obj.M = M;
            obj.Meff = Meff;
            obj.N = N;
            obj.R = R;
            obj.barGrid = barGrid;
            bar_durations = obj.meter_state2meter(1, :) ./ obj.meter_state2meter(2, :);
            obj.barGrid_eff = round(bar_durations .* obj.barGrid ./ max(bar_durations));
            obj.use_silence_state = use_silence_state;
            obj = obj.make_state2obs_idx;
            obj.feat_type = feat_type;
        end
                      
        params = fit_distribution(obj, data_file_pattern_barpos_dim)
         
        function obj = train_model(obj, train_data)
            % data_file_pattern_barpos_dim: cell [n_files x n_patterns x barpos x feat_dim]
            obj.learned_params = obj.fit_distribution(train_data.feats_file_pattern_barPos_dim);
            if obj.use_silence_state
                temp{1} = train_data.feats_silence;
                obj.learned_params(obj.R+1, 1) = obj.fit_distribution(temp);
            end
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
            % observations  : observations [nFrames x nDim]
            %
            %OUTPUT parameters:
            % obslik        : observation likelihood [R x barPos x nFrames]
            %
            % 21.03.2013 by Florian Krebs
            % ------------------------------------------------------------------------
            nFrames = size(observations, 1);
            obsLik = zeros(obj.R, obj.barGrid, nFrames);
            for iR = 1:obj.R
                barPos = obj.barGrid_eff(obj.rhythm2meter_state(iR));
                obsLik(iR, 1:barPos, :) = obj.lik_func_handle(observations, ...
                    obj.learned_params(iR, 1:barPos));
% 		temp = obsLik(iR, 1:barPos, :);
% 		temp(temp < eps) = eps;
% 		obsLik(iR, 1:barPos, :) = temp;
            end
            if obj.use_silence_state
                obsLik(obj.R+1, 1, 1:nFrames) = obj.lik_func_handle(observations, ...
                    obj.learned_params(obj.R+1, 1));
            end
        end
        
        lik_func_handle = set_lik_func_handle(obj)
        
        function mean_params = comp_mean_params(obj)
            % mean_params: [R x barGrid x featDim]
%             [R, barpos] = size(obj.learned_params);
            feat_dims = obj.learned_params{1, 1}.NDimensions;
            mean_params = zeros(obj.R, obj.barGrid, feat_dims);
            for iR=1:obj.R
               for b=1:obj.barGrid
                   if ~isempty(obj.learned_params{iR, b})
                        mean_params(iR, b, :)=obj.learned_params{iR, b}.PComponents * obj.learned_params{iR, b}.mu;
                   end
               end
            end           
        end
        
        function plot_learned_patterns(obj)
            mean_params = obj.comp_mean_params;
            h = figure;
            set(h, 'Position', [100 100 obj.R*100 obj.R*100]);
            plot_cols = ceil(sqrt(obj.R));
            col = hsv(obj.learned_params{1, 1}.NDimensions);
            for c = 1:obj.R
                subplot(ceil(obj.R/plot_cols), plot_cols, c)
                hold on
                for fdim = 1:obj.learned_params{1, 1}.NDimensions
                    data = mean_params(c, :, fdim);
                    data = data - min(data);
                    data = data / max(data);
                    data = data + fdim;
                    stairs([data, data(end)], 'Color', col(fdim, :));
%                     stairs([means(:, nDim-iDim+1); means(end, nDim-iDim+1)], 'k', 'LineWidth', 2)
                end
                title(sprintf('cluster %i', c));
                xlim([1 length(data)+1])
            end 
        end
        
        function obj = set_learned_params(obj, learned_params)
           obj.learned_params = learned_params; 
        end
        
        function obj = make_state2obs_idx(obj)
            %{
            Computes state2obs_idx, which specifies which states are tied (share the same parameters) 
            %}
            nStates = obj.M * obj.N * obj.R;
            obj.state2obs_idx = nan(nStates, 2);
            barPosPerGrid = obj.M / obj.barGrid;
            discreteBarPos = floor(((1:obj.M) - 1)/barPosPerGrid) + 1;
            for iR=1:obj.R
                Meff_iR = obj.Meff(obj.rhythm2meter_state(iR));
                r = ones(Meff_iR, 1) * iR;
                for iN = 1:obj.N
                    ind = sub2ind([obj.M, obj.N, obj.R], (1:Meff_iR)', repmat(iN, Meff_iR, 1), r);
                    obj.state2obs_idx(ind, 1) = r;
                    obj.state2obs_idx(ind, 2) = discreteBarPos(1:Meff_iR);
                end
            end
            if obj.use_silence_state
                obj.state2obs_idx(end+1, 1) = obj.R+1;
                obj.state2obs_idx(end, 2) = 1;
            end
            
        end 
    end
    
end
