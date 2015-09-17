classdef BeatTrackingObservationModel < handle
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        state_space
        feat_type
        dist_type
        cells_from_pattern
        max_cells
        compute_likelihood
        learned_params
    end
    
    methods
        function obj = BeatTrackingObservationModel(state_space, feat_type, ...
                dist_type, cells_per_whole_note)
            obj.state_space = state_space;
            obj.feat_type = feat_type;
            obj.dist_type = dist_type;
            bar_durations = obj.state_space.meter_from_pattern(:, 1) ./ ...
                obj.state_space.meter_from_pattern(:, 2);
            obj.cells_from_pattern = bar_durations * cells_per_whole_note;
            obj.max_cells = max(obj.cells_from_pattern);
            obj.set_likelihood_function_handle();
        end
        
        function obj = train_model(obj, train_data)
            features_by_clusters = train_data.sort_bars_into_clusters();
            obj.learned_params = obj.fit_distribution(features_by_clusters);
            if obj.state_space.use_silence_state
                silence{1} = train_data.feats_silence;
                obj.learned_params(obj.state_space.n_patterns + 1, 1) = ...
                    obj.fit_distribution(silence);
            end
        end
        
        function obj = retrain_model(obj, features_by_clusters, ...
                pattern_id)
            % update parameters of pattern_id
            obj.learned_params(pattern_id, :) = obj.fit_distribution(...
                features_by_clusters(:, pattern_id, :, :));
        end
        
        function obs_lik = compute_obs_lik(obj, observations)
            % obs_lik = compute_obs_lik(obj, observations)
            %
            % pre-computes the likelihood of the observations
            %
            % -----------------------------------------------------------------
            %INPUT parameters:
            % observations  : observations [nFrames x nDim]
            %
            %OUTPUT parameters:
            % obslik        : observation likelihood [R x barPos x nFrames]
            %
            % 27.08.2015 by Florian Krebs
            % -----------------------------------------------------------------
            n_frames = size(observations, 1);
            n_patterns = obj.state_space.n_patterns;
            obs_lik = ones(n_patterns, obj.max_cells, n_frames) * (-1);
            if isempty(obj.learned_params)
               % In cases where no parameters were learned, just pass zero 
               % matrix. This happens e.g., for the RNN observation likelihood.
               obj.learned_params = zeros(n_patterns + 1, obj.max_cells);
            end
            for r = 1:n_patterns
                max_pos_r = obj.cells_from_pattern(r);
                obs_lik(r, 1:max_pos_r, :) = obj.compute_likelihood...
                    (observations, obj.learned_params(r, 1:max_pos_r));
            end
            if obj.state_space.use_silence_state
                obs_lik(n_patterns + 1, 1, 1:n_frames) = ...
                    obj.compute_likelihood(observations, ...
                    obj.learned_params(n_patterns + 1, 1));
            end
        end
        
        function plot_learned_patterns(obj)
            means = obj.compute_distribution_means;
            n_patterns = obj.state_space.n_patterns;
            h = figure;
            set(h, 'Position', [100 100 n_patterns*200 n_patterns*200]);
            % how many axes next to each other?
            plot_cols = ceil(sqrt(n_patterns));
            colors = hsv(length(obj.feat_type));
            for c = 1:n_patterns
                subplot(ceil(n_patterns/plot_cols), plot_cols, c)
                hold on
                for fdim = 1:length(obj.feat_type)
                    data = means(c, :, fdim);
                    % normalise data to [0, 1] for better visualisation
                    data = data - min(data);
                    data = data / max(data);
                    data = data + (fdim - 1);
                    stairs(1:length(data)+1, [data, data(end)], 'Color', ...
                        colors(fdim, :));
                end
                title(sprintf('cluster %i', c));
                xlim([1 length(data)+1])
            end
        end
        
        Params = fit_distribution(obj, features_by_clusters);
        
    end
    
    methods (Access = protected)
        function dist_mean = compute_distribution_mean(obj, params)
            switch obj.dist_type
                case 'MOG'
                    dist_mean = params.PComponents * params.mu;
                otherwise
                    error('Cannot compute mean of distribution %s\n', ...
                        obj.dist_type);
            end
        end
        
        function mean_params = compute_distribution_means(obj)
        % Compute mean of distribution
            % mean_params: [n_patterns x max_cells x feat_dim]
            mean_params = zeros(obj.state_space.n_patterns, obj.max_cells, ...
                length(obj.feat_type));
            for r=1:obj.state_space.n_patterns
                n_cells = obj.cells_from_pattern(r);
                for b=1:n_cells
                    % TODO: vectorise
                    mean_params(r, b, :)= obj.compute_distribution_mean(...
                        obj.learned_params{r, b});
                end
            end
        end
        

        
        [] = set_likelihood_function_handle(obj)
        % set_likelihood_function_handle(obj)
        % Sets the function handle to compute the observation likelihood
        %
        % ----------------------------------------------------------------------
        %INPUT parameters:
        % obj
        %
        % 27.08.2015 by Florian Krebs
        % ----------------------------------------------------------------------
        
    end
    
end

