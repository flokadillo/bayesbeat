classdef ParticleFilter
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        state_space
        trans_model
        obs_model
        initial_particles
        n_particles
    end
    
    methods
        function obj = ParticleFilter(trans_model, obs_model, initial_particles, ...
                n_particles)
            obj.trans_model = trans_model;
            obj.obs_model = obs_model;
            obj.initial_particles = initial_particles;
            obj.n_particles = n_particles;
            obj.state_space = obj.trans_model.state_space;
        end
        
        function [m, n, r, w] = forward_filtering(obj, obs_lik)
            n_frames = size(obs_lik, 3);
            
            % initialize particles and preallocate memory
            m = zeros(obj.n_particles, n_frames, 'single');
            n = zeros(obj.n_particles, n_frames, 'single');
            r = zeros(obj.n_particles, n_frames, 'single');
            m(:, 1) = obj.initial_particles(:, 1);
            n(:, 1) = obj.initial_particles(:, 2);
            r(:, 1) = obj.initial_particles(:, 3);
            group_id = obj.initial_particles(:, 4);
            
            % use first observation
            obs = obj.likelihood_of_particles(m(:, 1), r(:, 1), ...
                obs_lik(:, :, 1));
            weight = log(obs / sum(obs));
            resampling_frames = zeros(n_frames, 1);
            
            for iFrame=2:n_frames
                % transition from iFrame-1 to iFrame
                m(:, iFrame) = obj.trans_model.update_position(m(:, iFrame-1), ...
                    n(:, iFrame-1), r(:, iFrame-1));
                r(:, iFrame) = obj.trans_model.sample_pattern(r(:, iFrame-1), ...
                    m(:, iFrame), m(:, iFrame-1), n(:, iFrame-1));
                % evaluate particle at iFrame-1
                obs = obj.likelihood_of_particles(m(:, iFrame), r(:, iFrame), ...
                    obs_lik(:, :, iFrame));
                weight = weight(:) + log(obs(:));
                % Normalise importance weights
                [weight, ~] = obj.normalizeLogspace(weight');
                % Resampling
                % ------------------------------------------------------------
                if obj.resampling_interval == 0
                    Neff = 1/sum(exp(weight).^2);
                    do_resampling = (Neff < obj.ratio_Neff * obj.n_particles);
                else
                    do_resampling = (rem(iFrame, obj.resampling_interval) == 0);
                end
                if do_resampling && (iFrame < n_frames)
                    resampling_frames(iFrame) = iFrame;
                    if obj.resampling_scheme == 2 || obj.resampling_scheme == 3 % MPF or AMPF
                        group_id = obj.divide_into_clusters([m(:, iFrame), ...
                            n(:, iFrame-1), r(:, iFrame)], ...
                            [obj.M; obj.N; obj.state_space.n_patterns], group_id);
                        n_clusters(iFrame) = length(unique(group_id));
                    end
                    [weight, group_id, newIdx] = obj.resample(weight, group_id);
                    m(:, 1:iFrame) = m(newIdx, 1:iFrame);
                    r(:, 1:iFrame) = r(newIdx, 1:iFrame);
                    n(:, 1:iFrame) = n(newIdx, 1:iFrame);
                end
                % transition from iFrame-1 to iFrame
                n(:, iFrame) = n(:, iFrame-1) + randn(obj.n_particles, 1) * obj.sigma_N * obj.M;
                out_of_range = n(:, iFrame)' > obj.maxN(r(:, iFrame));
                n(out_of_range, iFrame) = obj.maxN(r(out_of_range, iFrame));
                out_of_range = n(:, iFrame)' < obj.minN(r(:, iFrame));
                n(out_of_range, iFrame) = obj.minN(r(out_of_range, iFrame));
            end
            w = weight;
            obj.particles.weight = weight;
            fprintf('    Average resampling interval: %.2f frames\n', ...
                mean(diff(resampling_frames(resampling_frames>0))));
            if obj.resampling_scheme > 1
                fprintf('    Average number of clusters: %.2f frames\n', mean(n_clusters(n_clusters>0)));
            end
            if obj.save_inference_data
                save(['./', fname, '_pf.mat'], 'logP_data_pf');
            end
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
            m_path = m(best_last_particle, :);
            r_path = r(best_last_particle, :);
            n_path = n(best_last_particle, :)';
            m_path = m_path(:)';
            n_path = n_path(:)';
            r_path = r_path(:)';
        end
        
        
    end
    
    methods (Static)
        
        
    end
    
end

