classdef ParticleFilter
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        state_space
        trans_model
        obs_model
        initial_particles
        n_particles
        resampling_params
    end
    
    methods
        function obj = ParticleFilter(trans_model, obs_model, initial_particles, ...
                n_particles, resampling_params)
            obj.trans_model = trans_model;
            obj.obs_model = obs_model;
            obj.initial_particles = initial_particles;
            obj.n_particles = n_particles;
            obj.state_space = obj.trans_model.state_space;
            obj.resampling_params = resampling_params;
        end
        
        function [m, n, r, w] = forward_filtering(obj, obs_lik)
            % initialize particles and preallocate memory
            n_frames = size(obs_lik, 3);
            % add one frame, which corresponds to the initial distribution
            % at time 0 before the first observation comes in
            m = zeros(obj.n_particles, n_frames + 1, 'single');
            n = zeros(obj.n_particles, n_frames + 1, 'single');
            r = zeros(obj.n_particles, n_frames + 1, 'single');
            logP_data_pf = zeros(obj.n_particles, 5, n_frames, 'single');
            m(:, 1) = obj.initial_particles(:, 1);
            n(:, 1) = obj.initial_particles(:, 2);
            r(:, 1) = obj.initial_particles(:, 3);
            g = obj.initial_particles(:, 4);
            w = log(ones(obj.n_particles, 1) / obj.n_particles);   
            for iFrame=1:n_frames
                % sample position and pattern
                m(:, iFrame+1) = obj.trans_model.update_position(m(:, iFrame), ...
                    n(:, iFrame), r(:, iFrame));
                r(:, iFrame+1) = obj.trans_model.sample_pattern(r(:, iFrame), ...
                    m(:, iFrame+1), m(:, iFrame), n(:, iFrame));
                % evaluate likelihood of particles
                obs = obj.likelihood_of_particles(m(:, iFrame+1), r(:, iFrame+1), ...
                    obs_lik(:, :, iFrame));
                iFrame
                length(unique(m(:, iFrame+1)))
                if sum(obs) == 0
                    lkj=987;
                end
                    
                w = w(:) + log(obs(:));
                % normalise importance weights
                [w, ~] = obj.normalizeLogspace(w);
                % resampling
                if iFrame < n_frames
                    [m, n, r, g, w] = resampling(obj, m, n, r, g, w, iFrame);
                end
                if length(unique(m(:, iFrame+1))) < 100
                    a=8796;
                end
                % sample tempo after resampling because it has no impact on
                % the resampling and we achieve greater tempo diversity.
                n(:, iFrame+1) = obj.trans_model.sample_tempo(n(:, iFrame), ...
                    r(:, iFrame+1));
                logP_data_pf(:, 1, iFrame) = m(:, iFrame+1);
                logP_data_pf(:, 2, iFrame) = n(:, iFrame+1);
                logP_data_pf(:, 3, iFrame) = r(:, iFrame+1);
                logP_data_pf(:, 4, iFrame) = w;
                logP_data_pf(:, 5, iFrame) = g;
            end
            % remove initial state
            m = m(:, 2:end); n = n(:, 2:end); r = r(:, 2:end);
            save('/tmp/data_pf.mat', 'logP_data_pf', 'obs_lik');
        end
        
        function [m, n, r, g, w_log] = resampling(obj, m, n, r, g, w_log, iFrame)
            % compute reampling criterion effective sample size
            w = exp(w_log);
            Neff = 1/sum(w.^2);
            if Neff > (obj.resampling_params.ratio_Neff * obj.n_particles)
                return
            end
            if obj.resampling_params.resampling_scheme == 0 % SISR
                newIdx = obj.resampleSystematic(w);
                w_log = log(ones(obj.n_particles, 1) / obj.n_particles);
            elseif obj.resampling_params.resampling_scheme == 1 % APF
                % warping:
                w_warped = obj.resampling_params.warp_fun(w);
                newIdx = obj.resampleSystematic(w_warped);
                w_fac = w ./ w_warped;
                w_log = log( w_fac(newIdx) / sum(w_fac(newIdx)) );
            else
                fprintf('WARNING: Unknown resampling scheme!\n');
            end
            m(:, 1:iFrame) = m(newIdx, 1:iFrame);
            r(:, 1:iFrame) = r(newIdx, 1:iFrame);
            n(:, 1:iFrame) = n(newIdx, 1:iFrame);
        end
        
        
        function lik = likelihood_of_particles(obj, position, pattern, ...
                obs_lik)
            % obslik:       likelihood values [R, barGrid]
            m_per_grid = obj.state_space.max_position_from_pattern(1) / ...
                obj.obs_model.cells_from_pattern(1);
            p_cell = floor((position - 1) / m_per_grid) + 1;
            ind = sub2ind([obj.state_space.n_patterns, ...
                obj.obs_model.max_cells], pattern(:), p_cell(:));
            lik = obs_lik(ind);
        end
        
        function [m_path, n_path, r_path] = path_with_best_last_weight(obj, ...
                obs_lik)
            [m, n, r, w] = obj.forward_filtering(obs_lik);
            [~, best_last_particle] = max(w);
            m_path = m(best_last_particle, :)';
            r_path = r(best_last_particle, :)';
            n_path = n(best_last_particle, :)';
        end
        
        
    end
    
    methods (Static)
        function [y, L] = normalizeLogspace(x)
            % Normalize in logspace while avoiding numerical underflow
            % Each *row* of x is a log discrete distribution.
            % y(i,:) = x(i,:) - logsumexp(x,2) = x(i) - log[sum_c exp(x(i,c)]
            % L is the log normalization constant
            % eg [logPost, L] = normalizeLogspace(logprior + loglik)
            % post = exp(logPost);
            % This file is from pmtk3.googlecode.com
            x = x(:)';
            L = PF.logsumexp(x(:)', 2);
            y = bsxfun(@minus, x(:), L);
        end
        
        function r = logsumexp(X, dim)
            %LOG_SUM_EXP Numerically stable computation of log(sum(exp(X), dim))
            % [r] = log_sum_exp(X, dim)
            %
            % Inputs :
            %
            % X : Array
            %
            % dim : Sum Dimension <default = 1>, means summing over the columns
            % This file is from pmtk3.googlecode.com
            maxval = max(X,[],dim);
            sizes = size(X);
            if dim == 1
                normmat = repmat(maxval,sizes(1),1);
            else
                normmat = repmat(maxval,1,sizes(2));
            end
            r = maxval + log(sum(exp(X-normmat),dim));
        end
        
        function [ indx ] = resampleSystematic( w, n_samples )
            % n_samples ... number of samples after resampling
            w = w(:);
            if sum(w) > eps
                w = w / sum(w);
            else
                w = ones(size(w)) / length(w);
            end
            if exist('n_samples', 'var')
                N = n_samples;
            else
                N = length(w);
            end
            Q = cumsum(w);
            T = linspace(0,1-1/N,N) + rand(1)/N;
            T(N+1) = 1;
            [~, indx] = histc(T, [0; Q]);
            indx = indx(1:end-1);
        end
        
    end
end

